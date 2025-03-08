#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32, Bool

class SteeringJudgement:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('steering_judgement_node', anonymous=True)
        
        # 구독자 설정
        self.sub_lane = rospy.Subscriber('auto_steer_angle_lane', Float32, self.lane_callback)
        self.sub_cone = rospy.Subscriber('auto_steer_angle_cone', Float32, self.cone_callback)
        self.sub_tunnel = rospy.Subscriber('auto_steer_angle_tunnel', Float32, self.tunnel_callback)
        self.sub_lane_status = rospy.Subscriber('lane_detection_status', Bool, self.lane_status_callback)
        # 퍼블리셔 설정
        self.pub_steering = rospy.Publisher("steering_angle", Float32, queue_size=10)
        
        # 변수 초기화
        self.lane_angle = None
        self.cone_angle = None
        self.tunnel_angle = None
        self.lane_detected = False
        self.last_lane_time = rospy.Time(0)
        self.last_cone_time = rospy.Time(0)
        self.last_tunnel_time = rospy.Time(0)
        
        # 타이머 설정 (10Hz로 주기적으로 퍼블리시)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_steering)
        
        # 초기화 로그
        rospy.loginfo("SteeringJudgement node initialized and subscribing to topics.")

    def lane_callback(self, msg):
        self.lane_angle = msg.data
        self.last_lane_time = rospy.Time.now()
        rospy.loginfo("Received lane angle: %.2f", self.lane_angle)

    def cone_callback(self, msg):
        self.cone_angle = msg.data
        self.last_cone_time = rospy.Time.now()
        rospy.loginfo("Received cone angle: %.2f", self.cone_angle)

    def tunnel_callback(self, msg):
        self.tunnel_angle = msg.data
        self.last_tunnel_time = rospy.Time.now()
        rospy.loginfo("Received tunnel angle: %.2f", self.tunnel_angle)

    def lane_status_callback(self, msg):
        self.lane_detected = msg.data

    def publish_steering(self, event):
        steering_angle = None
        used_source = "None"

        # 우선순위: lane > cone > tunnel (최신 데이터 기준)
        if self.lane_detected and self.lane_angle is not None:
            steering_angle = self.lane_angle
            used_source = "auto_steer_angle_lane"
        else:
            if self.cone_angle is not None and self.tunnel_angle is not None:
                if self.last_cone_time > self.last_tunnel_time:
                    steering_angle = self.cone_angle
                    used_source = "auto_steer_angle_cone"
                else:
                    steering_angle = self.tunnel_angle
                    used_source = "auto_steer_angle_tunnel"
            elif self.cone_angle is not None:
                steering_angle = self.cone_angle
                used_source = "auto_steer_angle_cone"
            elif self.tunnel_angle is not None:
                steering_angle = self.tunnel_angle
                used_source = "auto_steer_angle_tunnel"

        # steering_angle이 있으면 퍼블리시 및 로그 출력
        if steering_angle is not None:
            self.pub_steering.publish(Float32(data=steering_angle))
            rospy.loginfo("Published steering_angle: %.2f using topic: %s", steering_angle, used_source)
        else:
            rospy.loginfo("No valid steering angle available to publish")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        steering_judgement = SteeringJudgement()
        steering_judgement.run()
    except rospy.ROSInterruptException:
        pass