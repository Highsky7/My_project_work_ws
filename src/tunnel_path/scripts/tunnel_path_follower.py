#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
import tf
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

class TunnelPathNode:
    def __init__(self):
        # 터널 여부 및 센서 데이터 초기화
        self.in_tunnel = False
        self.current_pose = None
        self.left_avg = None
        self.right_avg = None

        # 터널로 인식하기 위한 임계치 (m 단위)
        self.tunnel_threshold = 0.5

        # 퍼블리셔: 터널 구간 전용 조향각과 경로를 퍼블리시
        self.steer_pub = rospy.Publisher('auto_steer_angle_tunnel', Float32, queue_size=10)
        self.path_pub = rospy.Publisher('path', Path, queue_size=10)

        # 구독자: 라이다 센서(예: /scan)와 차량의 자세정보(예: /odom)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)

        # 주기적으로 터널 여부와 경로, 조향각을 계산하는 타이머 (0.1초 주기)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        # 순수추종에 필요한 파라미터들
        self.wheelbase = 0.75          # 차량의 축간 거리 (예: 0.5 m, 1/5 스케일 차량에 맞게 설정)
        self.lookahead_distance = 1.0   # Lookahead 거리 (m 단위)
        self.path_length = 5.0          # 생성할 경로의 총 길이 (m)
        self.path_step = 0.3            # 경로 생성 시 점 간 간격 (m)
        
        rospy.loginfo("Tunnel Path Node Initialized")

    def scan_callback(self, scan_msg):
        """
        라이다 센서의 LaserScan 메시지에서 좌우(약 ±90°)의 거리를 추출하여
        터널 구간 여부를 판단합니다.
        """
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        # ±90° 근방 (예: 80° ~ 100°와 -100° ~ -80°) 범위의 인덱스 산출
        left_min_rad = math.radians(80)
        left_max_rad = math.radians(100)
        right_min_rad = math.radians(-100)
        right_max_rad = math.radians(-80)

        left_indices = []
        right_indices = []
        
        # angle_min부터 angle_max까지 각도에 해당하는 인덱스를 확인
        for i, angle in enumerate(np.arange(angle_min, scan_msg.angle_max, angle_increment)):
            if left_min_rad <= angle <= left_max_rad:
                left_indices.append(i)
            if right_min_rad <= angle <= right_max_rad:
                right_indices.append(i)

        # 인덱스가 하나도 없으면 패스
        if not left_indices or not right_indices:
            return

        # 인덱스에 해당하는 거리값 중 무한대(inf)는 제외
        left_values = [scan_msg.ranges[i] for i in left_indices if not math.isinf(scan_msg.ranges[i])]
        right_values = [scan_msg.ranges[i] for i in right_indices if not math.isinf(scan_msg.ranges[i])]

        # 평균 거리를 계산 (값이 없으면 inf로 처리)
        if left_values:
            self.left_avg = sum(left_values) / len(left_values)
        else:
            self.left_avg = float('inf')

        if right_values:
            self.right_avg = sum(right_values) / len(right_values)
        else:
            self.right_avg = float('inf')

        # 좌우 모두 임계치보다 짧으면 터널 구간으로 판단
        if self.left_avg < self.tunnel_threshold and self.right_avg < self.tunnel_threshold:
            self.in_tunnel = True
        else:
            self.in_tunnel = False

    def odom_callback(self, odom_msg):
        """
        차량의 현재 자세 정보를 저장합니다.
        """
        self.current_pose = odom_msg.pose.pose

    def generate_path(self, offset):
        """
        현재 차량의 자세를 기준으로, 일정 간격으로 직진하는 경로를 생성합니다.
        좌우 벽과의 상대적 거리 차(offset)를 반영하여 경로를 좌우로 평행 이동시킵니다.
        경로는 nav_msgs/Path 타입으로 생성됩니다.
        """
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "odom"  # odom 또는 map frame을 사용

        # 생성할 경로의 점 개수 계산
        num_points = int(self.path_length / self.path_step)
        
        # 차량의 현재 yaw(방향) 계산
        if self.current_pose is None:
            return path
        quaternion = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]

        # 경로 생성 (차량 좌표계 기준: x축 방향으로 일정 간격, y값은 offset 적용)
        for i in range(1, num_points + 1):
            ps = PoseStamped()
            ps.header.stamp = rospy.Time.now()
            ps.header.frame_id = "odom"
            
            # 차량 좌표계 상의 점 (직진 경로)
            x_local = i * self.path_step
            y_local = offset  # offset을 적용하여 좌우 이동
            
            # 현재 차량 위치와 yaw를 기준으로 전역 좌표로 변환
            ps.pose.position.x = self.current_pose.position.x + math.cos(yaw) * x_local - math.sin(yaw) * y_local
            ps.pose.position.y = self.current_pose.position.y + math.sin(yaw) * x_local + math.cos(yaw) * y_local
            ps.pose.position.z = self.current_pose.position.z
            
            # 차량의 방향과 동일한 orientation 적용
            quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
            ps.pose.orientation.x = quat[0]
            ps.pose.orientation.y = quat[1]
            ps.pose.orientation.z = quat[2]
            ps.pose.orientation.w = quat[3]
            
            path.poses.append(ps)
        return path

    def compute_steering_angle(self, offset):
        """
        순수추종(Pure Pursuit) 알고리즘을 이용해 조향각을 계산합니다.
        차량 좌표계 상의 lookahead 지점은 (L_d, offset)로 가정하며,
        조향각 δ는 다음과 같이 계산합니다.
            δ = arctan( 2 * wheelbase * offset / (L_d)^2 )
        """
        Ld = self.lookahead_distance
        delta = math.atan2(2 * self.wheelbase * offset, Ld**2)
        return delta

    def timer_callback(self, event):
        """
        주기적 타이머 콜백 함수에서 터널 구간인지 확인한 후,
        터널 구간이면 경로와 조향각을 계산하여 퍼블리시합니다.
        터널이 아닌 구간에서는 퍼블리시하지 않습니다.
        """
        if self.in_tunnel and self.current_pose is not None:
            # 좌우 평균 거리의 차이를 이용해 차량의 중앙 복귀를 위한 offset 산출
            # (왼쪽 벽에 너무 가까우면 오른쪽으로, 반대의 경우 왼쪽으로)
            if self.left_avg is not None and self.right_avg is not None:
                offset = (self.right_avg - self.left_avg) / 2.0
            else:
                offset = 0.0

            # 경로와 조향각 계산
            path_msg = self.generate_path(offset)
            steering_angle = self.compute_steering_angle(offset)
            
            # 퍼블리시
            steer_msg = Float32()
            steer_msg.data = steering_angle
            self.steer_pub.publish(steer_msg)
            self.path_pub.publish(path_msg)
            rospy.loginfo("터널 구간: 조향각 {:.3f} rad 퍼블리시, offset: {:.3f}".format(steering_angle, offset))
        else:
            rospy.loginfo("터널 구간이 아님. 퍼블리시하지 않음.")

if __name__ == '__main__':
    rospy.init_node('tunnel_path_node', anonymous=True)
    node = TunnelPathNode()
    rospy.spin()
