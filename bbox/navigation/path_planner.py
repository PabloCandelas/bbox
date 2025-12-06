#!/usr/bin/env python3
"""
path_planner.py - Path Planning for BlueROV2
=============================================

WHAT IS PATH PLANNING?
----------------------
Path planning answers the question: "How do I get from point A to point B
without hitting anything?"

For your underwater SAR mission, the robot needs to:
1. Search the pool area systematically to find the blackbox
2. Navigate to the blackbox while avoiding obstacles
3. Return to the surface/home position with the blackbox

TYPES OF PATH PLANNING:
-----------------------

1. COVERAGE PLANNING (Search Pattern):
   - "Lawn mower" pattern to systematically search the pool
   - Ensures the camera sees every part of the pool floor
   
2. POINT-TO-POINT PLANNING:
   - Navigate from current position to a goal
   - Avoid known obstacles
   
3. REACTIVE/LOCAL PLANNING:
   - Handle unexpected obstacles detected in real-time
   - Small corrections to avoid collisions

ALGORITHMS IMPLEMENTED:
-----------------------

1. GRID-BASED A* (A-Star):
   - Classic pathfinding algorithm
   - Discretizes space into a grid
   - Finds shortest path that avoids obstacles
   - Good for: Known obstacle maps

2. COVERAGE PATTERN:
   - Boustrophedon (back-and-forth) pattern
   - Good for: Searching the pool systematically

3. SIMPLE WAYPOINT GENERATION:
   - Direct paths when no obstacles
   - Good for: Open water navigation

YOUR POOL SETUP:
----------------
- Pool dimensions: approximately 9m x 8m x 5m (based on marker positions)
- Markers at Z = 4.8m (pool floor)
- X range: 1.5 to 7.5m
- Y range: 1.0 to 7.0m
- Obstacles: Unknown positions (will be detected by sonar/camera)
"""

import numpy as np
import math
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq

from .transforms import Pose3D, Transforms


class PlanningMode(Enum):
    """Different planning modes for different tasks."""
    SEARCH = 1      # Coverage search pattern
    NAVIGATE = 2    # Point-to-point navigation
    APPROACH = 3    # Careful approach to target


@dataclass
class Obstacle:
    """
    Represents an obstacle in the environment.
    
    Obstacles are modeled as spheres or cylinders for simplicity.
    
    Attributes:
        center: (x, y, z) position in world frame
        radius: Radius of the obstacle (meters)
        height: Height for cylindrical obstacles (None = sphere)
        dynamic: True if obstacle might move
    """
    center: np.ndarray
    radius: float = 0.5
    height: Optional[float] = None
    dynamic: bool = False
    
    def contains_point(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if a point is inside the obstacle (with optional margin)."""
        if self.height is None:
            # Sphere check
            dist = np.linalg.norm(point - self.center)
            return dist < (self.radius + margin)
        else:
            # Cylinder check
            xy_dist = np.linalg.norm(point[:2] - self.center[:2])
            z_in_range = (self.center[2] <= point[2] <= self.center[2] + self.height)
            return xy_dist < (self.radius + margin) and z_in_range


@dataclass
class PlannerConfig:
    """Configuration for the path planner."""
    
    # Pool boundaries (in world frame)
    pool_x_min: float = 0.5
    pool_x_max: float = 8.5
    pool_y_min: float = 0.5
    pool_y_max: float = 7.5
    pool_z_min: float = 0.5   # Minimum depth (don't break surface)
    pool_z_max: float = 4.5   # Maximum depth (don't hit bottom)
    
    # Robot parameters
    robot_radius: float = 0.3  # Safety radius around robot (meters)
    
    # Search pattern parameters
    search_altitude: float = 2.5   # Depth for search pattern
    search_spacing: float = 1.5    # Spacing between search lines (meters)
    search_overlap: float = 0.3    # Overlap ratio between lines
    
    # Grid parameters (for A*)
    grid_resolution: float = 0.25  # Grid cell size (meters)
    
    # Path smoothing
    smooth_path: bool = True
    max_turn_angle: float = math.pi / 4  # Maximum turn angle (radians)


@dataclass
class Waypoint:
    """
    A waypoint in the path.
    
    Attributes:
        position: (x, y, z) in world frame
        yaw: Desired heading (radians, optional)
        speed: Desired speed to reach this waypoint
        action: Special action at this waypoint (e.g., 'hover', 'look_down')
    """
    position: np.ndarray
    yaw: Optional[float] = None
    speed: float = 0.3  # m/s
    action: str = ''
    
    def to_pose(self) -> Pose3D:
        """Convert waypoint to Pose3D."""
        yaw = self.yaw if self.yaw is not None else 0.0
        q = Transforms.quaternion_from_euler(0, 0, yaw)
        return Pose3D(
            x=self.position[0], y=self.position[1], z=self.position[2],
            qx=q[0], qy=q[1], qz=q[2], qw=q[3]
        )


class PathPlanner:
    """
    Path planner for BlueROV2.
    
    This class generates paths for:
    1. Search patterns to find the blackbox
    2. Point-to-point navigation
    3. Approach paths to targets
    
    Usage:
        planner = PathPlanner()
        
        # Generate search pattern
        search_path = planner.generate_search_pattern()
        
        # Navigate to a goal
        path = planner.plan_path(current_pose, goal_pose)
        
        # Add an obstacle
        planner.add_obstacle(Obstacle(center=np.array([3, 4, 2]), radius=0.5))
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """Initialize the path planner."""
        self.config = config or PlannerConfig()
        self.obstacles: List[Obstacle] = []
        
        # Pre-compute pool dimensions
        self.pool_width = self.config.pool_x_max - self.config.pool_x_min
        self.pool_length = self.config.pool_y_max - self.config.pool_y_min
    
    # =========================================================================
    # OBSTACLE MANAGEMENT
    # =========================================================================
    
    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the map."""
        self.obstacles.append(obstacle)
    
    def remove_obstacle(self, index: int) -> None:
        """Remove an obstacle by index."""
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles.clear()
    
    def is_collision_free(self, point: np.ndarray) -> bool:
        """
        Check if a point is collision-free.
        
        Args:
            point: (x, y, z) position to check
            
        Returns:
            True if the point is safe (no collision)
        """
        # Check pool boundaries
        if not self._in_bounds(point):
            return False
        
        # Check against all obstacles
        for obs in self.obstacles:
            if obs.contains_point(point, margin=self.config.robot_radius):
                return False
        
        return True
    
    def _in_bounds(self, point: np.ndarray) -> bool:
        """Check if a point is within pool boundaries."""
        x, y, z = point
        return (self.config.pool_x_min <= x <= self.config.pool_x_max and
                self.config.pool_y_min <= y <= self.config.pool_y_max and
                self.config.pool_z_min <= z <= self.config.pool_z_max)
    
    # =========================================================================
    # SEARCH PATTERN GENERATION
    # =========================================================================
    
    def generate_search_pattern(self, 
                                start_corner: str = 'bottom_left',
                                altitude: Optional[float] = None) -> List[Waypoint]:
        """
        Generate a lawn-mower search pattern to cover the pool.
        
        This creates a back-and-forth (boustrophedon) pattern that ensures
        the camera sees every part of the pool floor.
        
        Args:
            start_corner: Where to start ('bottom_left', 'bottom_right', 
                         'top_left', 'top_right')
            altitude: Depth for the search pattern (default: config value)
            
        Returns:
            List of Waypoints forming the search pattern
            
        Visual representation (top view):
        
            START → ──────────┐
                              │
            ┌─────────────────┘
            │
            └─────────────────┐
                              │
            END ← ────────────┘
        """
        if altitude is None:
            altitude = self.config.search_altitude
        
        waypoints = []
        
        # Calculate effective spacing (considering overlap)
        spacing = self.config.search_spacing * (1 - self.config.search_overlap)
        
        # Determine start and end X coordinates
        x_start = self.config.pool_x_min + self.config.robot_radius
        x_end = self.config.pool_x_max - self.config.robot_radius
        
        y_start = self.config.pool_y_min + self.config.robot_radius
        y_end = self.config.pool_y_max - self.config.robot_radius
        
        # Generate Y coordinates for each line
        y_positions = []
        y = y_start
        while y <= y_end:
            y_positions.append(y)
            y += spacing
        
        # Add final line if not at edge
        if y_positions[-1] < y_end - spacing * 0.5:
            y_positions.append(y_end)
        
        # Generate the pattern
        going_right = (start_corner in ['bottom_left', 'top_left'])
        
        for i, y in enumerate(y_positions):
            if going_right:
                # Left to right
                waypoints.append(Waypoint(
                    position=np.array([x_start, y, altitude]),
                    yaw=0.0,  # Facing positive X
                    action='search'
                ))
                waypoints.append(Waypoint(
                    position=np.array([x_end, y, altitude]),
                    yaw=0.0,
                    action='search'
                ))
            else:
                # Right to left
                waypoints.append(Waypoint(
                    position=np.array([x_end, y, altitude]),
                    yaw=math.pi,  # Facing negative X
                    action='search'
                ))
                waypoints.append(Waypoint(
                    position=np.array([x_start, y, altitude]),
                    yaw=math.pi,
                    action='search'
                ))
            
            # Alternate direction
            going_right = not going_right
        
        return waypoints
    
    def generate_spiral_search(self,
                               center: np.ndarray,
                               max_radius: float = 3.0,
                               altitude: Optional[float] = None) -> List[Waypoint]:
        """
        Generate an expanding spiral search pattern.
        
        Useful when you have a rough idea of where the target is.
        
        Args:
            center: (x, y, z) center point to spiral around
            max_radius: Maximum spiral radius
            altitude: Depth for the search
            
        Returns:
            List of Waypoints forming the spiral
        """
        if altitude is None:
            altitude = self.config.search_altitude
        
        waypoints = []
        
        # Spiral parameters
        points_per_turn = 12
        spacing = self.config.search_spacing * 0.8
        
        # Generate spiral points
        angle = 0
        radius = 0.3  # Start close to center
        
        while radius < max_radius:
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            
            # Check bounds
            point = np.array([x, y, altitude])
            if self._in_bounds(point):
                # Calculate yaw to point towards center of spiral
                yaw_to_center = math.atan2(center[1] - y, center[0] - x)
                
                waypoints.append(Waypoint(
                    position=point,
                    yaw=yaw_to_center,
                    action='search'
                ))
            
            # Advance spiral
            angle += 2 * math.pi / points_per_turn
            radius += spacing / points_per_turn
        
        return waypoints
    
    # =========================================================================
    # POINT-TO-POINT PATH PLANNING (A* Algorithm)
    # =========================================================================
    
    def plan_path(self, 
                  start: Pose3D, 
                  goal: Pose3D,
                  allow_3d: bool = True) -> List[Waypoint]:
        """
        Plan a collision-free path from start to goal.
        
        Uses the A* algorithm on a 3D grid.
        
        Args:
            start: Starting pose
            goal: Goal pose
            allow_3d: If True, can plan in 3D. If False, keeps current depth.
            
        Returns:
            List of Waypoints from start to goal, or empty list if no path found.
            
        How A* works (for beginners):
        1. Start at beginning, mark it as "visited"
        2. Look at all neighbors, calculate cost to reach them
        3. Pick the neighbor with lowest (cost_so_far + estimated_cost_to_goal)
        4. Repeat until we reach the goal
        5. Trace back to get the path
        """
        start_pos = start.position()
        goal_pos = goal.position()
        
        if not allow_3d:
            goal_pos[2] = start_pos[2]
        
        # Check if direct path is possible (no obstacles)
        if self._is_line_clear(start_pos, goal_pos):
            # Direct path is clear - just return start and goal
            return self._create_direct_path(start, goal)
        
        # Need to use A* to find path around obstacles
        path_positions = self._astar_3d(start_pos, goal_pos)
        
        if not path_positions:
            return []  # No path found
        
        # Convert to waypoints
        waypoints = []
        goal_yaw = Transforms.euler_from_quaternion(goal.quaternion())[2]
        
        for i, pos in enumerate(path_positions):
            # Calculate yaw to face next waypoint
            if i < len(path_positions) - 1:
                next_pos = path_positions[i + 1]
                yaw = math.atan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
            else:
                yaw = goal_yaw
            
            waypoints.append(Waypoint(position=pos, yaw=yaw))
        
        # Smooth the path if enabled
        if self.config.smooth_path:
            waypoints = self._smooth_path(waypoints)
        
        return waypoints
    
    def _astar_3d(self, 
                  start: np.ndarray, 
                  goal: np.ndarray) -> List[np.ndarray]:
        """
        A* pathfinding algorithm on a 3D grid.
        
        Args:
            start: Starting position
            goal: Goal position
            
        Returns:
            List of positions from start to goal
        """
        res = self.config.grid_resolution
        
        # Convert to grid coordinates
        def to_grid(pos):
            return tuple(int(p / res) for p in pos)
        
        def to_world(grid_pos):
            return np.array([g * res + res/2 for g in grid_pos])
        
        start_grid = to_grid(start)
        goal_grid = to_grid(goal)
        
        # Heuristic: Euclidean distance
        def heuristic(a, b):
            return math.sqrt(sum((a[i] - b[i])**2 for i in range(3))) * res
        
        # Priority queue: (f_score, counter, grid_pos)
        counter = 0
        open_set = [(heuristic(start_grid, goal_grid), counter, start_grid)]
        came_from = {}
        
        g_score = {start_grid: 0}
        f_score = {start_grid: heuristic(start_grid, goal_grid)}
        
        # 26-connected neighbors (all adjacent cells in 3D)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx != 0 or dy != 0 or dz != 0:
                        dist = math.sqrt(dx**2 + dy**2 + dz**2)
                        neighbors.append((dx, dy, dz, dist))
        
        visited = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal_grid:
                # Reconstruct path
                path = [to_world(current)]
                while current in came_from:
                    current = came_from[current]
                    path.append(to_world(current))
                return path[::-1]
            
            for dx, dy, dz, dist in neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                
                if neighbor in visited:
                    continue
                
                # Check if neighbor is valid
                neighbor_world = to_world(neighbor)
                if not self.is_collision_free(neighbor_world):
                    continue
                
                tentative_g = g_score[current] + dist * res
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal_grid)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
        
        return []  # No path found
    
    def _is_line_clear(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if a straight line path is collision-free."""
        # Sample points along the line
        distance = np.linalg.norm(end - start)
        num_samples = max(2, int(distance / self.config.grid_resolution))
        
        for i in range(num_samples + 1):
            t = i / num_samples
            point = start + t * (end - start)
            if not self.is_collision_free(point):
                return False
        
        return True
    
    def _create_direct_path(self, start: Pose3D, goal: Pose3D) -> List[Waypoint]:
        """Create a simple direct path from start to goal."""
        start_pos = start.position()
        goal_pos = goal.position()
        goal_yaw = Transforms.euler_from_quaternion(goal.quaternion())[2]
        
        # Calculate yaw to face the goal
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        travel_yaw = math.atan2(dy, dx)
        
        return [
            Waypoint(position=start_pos.copy(), yaw=travel_yaw),
            Waypoint(position=goal_pos.copy(), yaw=goal_yaw)
        ]
    
    def _smooth_path(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """
        Smooth a path by removing unnecessary waypoints.
        
        If we can go directly from A to C without hitting obstacles,
        we don't need waypoint B.
        """
        if len(waypoints) <= 2:
            return waypoints
        
        smoothed = [waypoints[0]]
        
        i = 0
        while i < len(waypoints) - 1:
            # Try to skip ahead as far as possible
            furthest = i + 1
            for j in range(i + 2, len(waypoints)):
                if self._is_line_clear(waypoints[i].position, waypoints[j].position):
                    furthest = j
            
            # Add the furthest reachable waypoint
            smoothed.append(waypoints[furthest])
            i = furthest
        
        # Recalculate yaw angles
        for i in range(len(smoothed) - 1):
            pos = smoothed[i].position
            next_pos = smoothed[i + 1].position
            smoothed[i].yaw = math.atan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
        
        return smoothed
    
    # =========================================================================
    # APPROACH PATH GENERATION
    # =========================================================================
    
    def plan_approach(self,
                      current_pose: Pose3D,
                      target_position: np.ndarray,
                      approach_distance: float = 1.0,
                      approach_direction: Optional[np.ndarray] = None) -> List[Waypoint]:
        """
        Plan an approach path to a target.
        
        Creates a path that stops at a safe distance from the target,
        facing towards it.
        
        Args:
            current_pose: Current robot pose
            target_position: (x, y, z) of the target (e.g., blackbox)
            approach_distance: Distance to stop from target (meters)
            approach_direction: Optional preferred direction to approach from
            
        Returns:
            List of Waypoints for the approach
        """
        current_pos = current_pose.position()
        
        if approach_direction is None:
            # Approach from current position
            direction = current_pos - target_position
            direction = direction / (np.linalg.norm(direction) + 1e-6)
        else:
            direction = approach_direction / (np.linalg.norm(approach_direction) + 1e-6)
        
        # Calculate approach point
        approach_point = target_position + direction * approach_distance
        
        # Calculate yaw to face target
        dx = target_position[0] - approach_point[0]
        dy = target_position[1] - approach_point[1]
        approach_yaw = math.atan2(dy, dx)
        
        # Create approach pose
        q = Transforms.quaternion_from_euler(0, 0, approach_yaw)
        approach_pose = Pose3D(
            x=approach_point[0], y=approach_point[1], z=approach_point[2],
            qx=q[0], qy=q[1], qz=q[2], qw=q[3]
        )
        
        # Plan path to approach point
        path = self.plan_path(current_pose, approach_pose)
        
        if path:
            # Mark last waypoint as approach waypoint
            path[-1].action = 'approach_ready'
        
        return path
    
    def get_home_position(self) -> Waypoint:
        """
        Get the home/surface position.
        
        This is where the robot should return after grabbing the blackbox.
        """
        # Default home position: center of pool, near surface
        center_x = (self.config.pool_x_min + self.config.pool_x_max) / 2
        center_y = (self.config.pool_y_min + self.config.pool_y_max) / 2
        
        return Waypoint(
            position=np.array([center_x, center_y, self.config.pool_z_min + 0.5]),
            yaw=0.0,
            action='home'
        )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the path planner module."""
    
    print("=" * 60)
    print("PATH PLANNER MODULE TEST")
    print("=" * 60)
    
    planner = PathPlanner()
    
    # Test 1: Generate search pattern
    print("\n1. Search Pattern Generation:")
    search_path = planner.generate_search_pattern()
    print(f"   Generated {len(search_path)} waypoints")
    print(f"   First waypoint: {search_path[0].position}")
    print(f"   Last waypoint: {search_path[-1].position}")
    
    # Test 2: Add obstacle and plan path
    print("\n2. Path Planning with Obstacle:")
    planner.add_obstacle(Obstacle(center=np.array([4, 4, 2]), radius=1.0))
    
    start = Pose3D(x=1, y=1, z=2)
    goal = Pose3D(x=7, y=7, z=2)
    
    path = planner.plan_path(start, goal)
    print(f"   Path found: {len(path) > 0}")
    print(f"   Number of waypoints: {len(path)}")
    
    if path:
        print(f"   Start: {path[0].position}")
        print(f"   Goal: {path[-1].position}")
    
    # Test 3: Spiral search
    print("\n3. Spiral Search Pattern:")
    spiral = planner.generate_spiral_search(center=np.array([4.5, 4, 2.5]))
    print(f"   Generated {len(spiral)} waypoints")
    
    # Test 4: Approach path
    print("\n4. Approach Path to Target:")
    planner.clear_obstacles()
    current = Pose3D(x=2, y=2, z=2)
    target = np.array([5, 5, 2])
    
    approach_path = planner.plan_approach(current, target, approach_distance=1.0)
    print(f"   Approach path length: {len(approach_path)}")
    if approach_path:
        print(f"   Final position: {approach_path[-1].position}")
        print(f"   Final yaw: {math.degrees(approach_path[-1].yaw):.1f}°")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
