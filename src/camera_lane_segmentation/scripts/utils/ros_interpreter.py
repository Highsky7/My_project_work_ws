#!/usr/bin/env python3
import sys
import rospy

def main():
    rospy.init_node('check_python_interpreter', anonymous=True)
    # 현재 사용 중인 Python 인터프리터의 전체 경로 출력
    rospy.loginfo("Using Python interpreter: %s", sys.executable)
    # Python 버전도 함께 출력하여 확인
    rospy.loginfo("Python version: %s", sys.version)
    
    # 노드가 바로 종료되지 않도록 spin() 호출
    rospy.spin()

if __name__ == '__main__':
    main()
