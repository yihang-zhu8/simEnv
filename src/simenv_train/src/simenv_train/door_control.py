"""
门和电梯控制

比赛允许使用 /set_door_state 和 /call_elevator 服务。
"""
import rospy
from building_generator_interfaces.srv import SetDoorState, CallElevator


class DoorController:
    """门控制"""

    def __init__(self):
        rospy.wait_for_service('/set_door_state', timeout=10.0)
        self.set_door = rospy.ServiceProxy('/set_door_state', SetDoorState)

    def open(self, door_id: str):
        """打开指定门"""
        try:
            resp = self.set_door(door_id=door_id, open=True)
            return resp.accepted
        except rospy.ServiceException as e:
            rospy.logwarn(f"开门失败 {door_id}: {e}")
            return False

    def close(self, door_id: str):
        """关闭指定门"""
        try:
            resp = self.set_door(door_id=door_id, open=False)
            return resp.accepted
        except rospy.ServiceException as e:
            rospy.logwarn(f"关门失败 {door_id}: {e}")
            return False


class ElevatorController:
    """电梯控制"""

    def __init__(self):
        rospy.wait_for_service('/call_elevator', timeout=10.0)
        self.call_elevator = rospy.ServiceProxy('/call_elevator', CallElevator)

    def go_to_floor(self, elevator_id: str, floor: int, open_doors: bool = False):
        """呼叫电梯到指定楼层"""
        try:
            resp = self.call_elevator(
                elevator_id=elevator_id,
                target_floor=floor,
                open_doors=open_doors
            )
            return resp.accepted
        except rospy.ServiceException as e:
            rospy.logwarn(f"电梯呼叫失败 {elevator_id} floor {floor}: {e}")
            return False
