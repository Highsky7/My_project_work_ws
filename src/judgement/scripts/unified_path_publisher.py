#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

class UnifiedPathPublisher:
    def __init__(self):
        # Initialize node
        rospy.init_node('unified_path_publisher', anonymous=True)

        # Class variables to store latest messages
        self.lane_path = None
        self.final_waypoints = None
        self.lane_detected = False

        # Subscribers
        self.lane_path_sub = rospy.Subscriber("lane_path", Path, self.lane_path_callback)
        self.final_waypoints_sub = rospy.Subscriber("/final_waypoints", Path, self.final_waypoints_callback)
        self.lane_status_sub = rospy.Subscriber("lane_detection_status", Bool, self.lane_status_callback)

        # Publisher
        self.unified_path_pub = rospy.Publisher("/unified_path", Path, queue_size=10)

        # Timer to publish unified path at 10 Hz
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_unified_path)

        rospy.loginfo("Unified Path Publisher initialized.")

    def lane_path_callback(self, msg):
        self.lane_path = msg

    def final_waypoints_callback(self, msg):
        self.final_waypoints = msg

    def lane_status_callback(self, msg):
        self.lane_detected = msg.data

    def publish_unified_path(self, event):
        # Skip if no final_waypoints available (required baseline)
        if self.final_waypoints is None or not self.final_waypoints.poses:
            return

        unified_path = Path()
        unified_path.header = self.final_waypoints.header  # Use "velodyne" frame

        if self.lane_detected and self.lane_path is not None and self.lane_path.poses:
            # Use lane_path fully
            unified_path.poses.extend(self.lane_path.poses)
            last_lane_point = self.lane_path.poses[-1].pose.position

            # Find closest point in final_waypoints
            min_dist = float('inf')
            closest_idx = -1
            for idx, pose in enumerate(self.final_waypoints.poses):
                p = pose.pose.position
                dist = (p.x - last_lane_point.x)**2 + (p.y - last_lane_point.y)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx

            if closest_idx != -1:
                closest_point = self.final_waypoints.poses[closest_idx].pose.position
                # Linear interpolation (5 points) for smooth transition
                num_interp = 5
                for i in range(1, num_interp):
                    t = i / float(num_interp)
                    x = last_lane_point.x + t * (closest_point.x - last_lane_point.x)
                    y = last_lane_point.y + t * (closest_point.y - last_lane_point.y)
                    ps = PoseStamped()
                    ps.header = unified_path.header
                    ps.pose.position.x = x
                    ps.pose.position.y = y
                    ps.pose.position.z = 0.0
                    ps.pose.orientation.w = 1.0
                    unified_path.poses.append(ps)
                # Append remaining final_waypoints
                unified_path.poses.extend(self.final_waypoints.poses[closest_idx:])
                rospy.loginfo("Published unified path: lane_path + transition + final_waypoints")
            else:
                rospy.logwarn("No close point found in final_waypoints.")
        else:
            # Use final_waypoints only
            unified_path.poses.extend(self.final_waypoints.poses)
            rospy.loginfo("Published unified path: final_waypoints only")

        self.unified_path_pub.publish(unified_path)

if __name__ == '__main__':
    try:
        node = UnifiedPathPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass