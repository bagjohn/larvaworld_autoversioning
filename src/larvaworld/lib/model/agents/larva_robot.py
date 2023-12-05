from ... import aux
from . import LarvaSim
from ...model.modules.motor_controller import MotorController, Actuator
from ...model.modules.sensor2 import ProximitySensor
from ...param import PositiveNumber, PositiveInteger


__all__ = [
    'LarvaRobot',
    'ObstacleLarvaRobot',
]

__displayname__ = 'Braitenberg-like larva'

class LarvaRobot(LarvaSim):

    def __init__(self, larva_pars,genome=None,**kwargs):
        super().__init__(**larva_pars, **kwargs)
        self.genome = genome



class ObstacleLarvaRobot(LarvaRobot):
    sensor_delta_direction = PositiveNumber(0.4, doc='Sensor delta_direction')
    sensor_saturation_value = PositiveNumber(40.0, doc='Sensor saturation value')
    obstacle_sensor_error = PositiveNumber(0.35, doc='Proximity sensor error')
    sensor_max_distance = PositiveNumber(0.9, doc='Sensor max_distance')
    motor_ctrl_coefficient = PositiveNumber(8770.0, doc='Motor ctrl_coefficient')
    motor_ctrl_min_actuator_value = PositiveNumber(35.0, doc='Motor ctrl_min_actuator_value')


    def __init__(self, larva_pars, **kwargs):
        kws = larva_pars.sensorimotor
        larva_pars.pop('sensorimotor', None)
        super().__init__(larva_pars=larva_pars, **kws,**kwargs)
        self.left_motor_controller = None
        self.right_motor_controller = None
        self.build_sensorimotor()

    def build_sensorimotor(self):

        S_kws = {
            'saturation_value': self.sensor_saturation_value,
            'error': self.obstacle_sensor_error,
            'max_distance': int(self.model.screen_manager.viewer._scale[0, 0] * self.sensor_max_distance * self.length),
            'collision_distance': int(self.model.screen_manager.viewer._scale[0, 0] * self.length / 5),
        }

        M_kws = {
            'coefficient': self.motor_ctrl_coefficient,
            'min_actuator_value': self.motor_ctrl_min_actuator_value,
        }

        Lsens = ProximitySensor(self, delta_direction=self.sensor_delta_direction, **S_kws)
        Rsens = ProximitySensor(self, delta_direction=-self.sensor_delta_direction, **S_kws)
        Lact = Actuator()
        Ract = Actuator()
        Lmot = MotorController(sensor=Lsens, actuator=Lact, **M_kws)
        Rmot = MotorController(sensor=Rsens, actuator=Ract, **M_kws)

        self.set_left_motor_controller(Lmot)
        self.set_right_motor_controller(Rmot)

    def sense(self):
        if not self.collision_with_object:
            pos = self.model.screen_manager.viewer._transform(self.olfactor_pos)
            try:

                self.left_motor_controller.sense_and_act(pos=pos, direction=self.direction)
                self.right_motor_controller.sense_and_act(pos=pos, direction=self.direction)
                Ltorque = self.left_motor_controller.get_actuator_value()
                Rtorque = self.right_motor_controller.get_actuator_value()
                dRL = Rtorque - Ltorque
                if dRL > 0:
                    self.brain.locomotor.turner.neural_oscillator.E_r += dRL * self.model.dt
                else:
                    self.brain.locomotor.turner.neural_oscillator.E_l -= dRL * self.model.dt
                # ang=self.head.get_angularvelocity()
                # self.head.set_ang_vel(ang-dRL*self.model.dt)
                # if dRL!=0 :
                #
                #     print(dRL*self.model.dt, ang)
                # self.brain.locomotor.turner.neural_oscillator.E_r += Rtorque * self.model.dt
                # self.brain.locomotor.turner.neural_oscillator.E_l += Ltorque * self.model.dt
                # if dRL>0 :
                #     self.brain.locomotor.turner.neural_oscillator.E_r += np.abs(dRL)
                # else :
                #     self.brain.locomotor.turner.neural_oscillator.E_l += np.abs(dRL)
            except aux.Collision:
                self.collision_with_object = True
                self.brain.locomotor.intermitter.interrupt_locomotion()
        else:
            pass

    def set_left_motor_controller(self, left_motor_controller):
        self.left_motor_controller = left_motor_controller

    def set_right_motor_controller(self, right_motor_controller):
        self.right_motor_controller = right_motor_controller

    def draw(self, v, **kwargs):
        # pos = self.olfactor_pos
        pos = v._transform(self.olfactor_pos)
        # draw the sensor lines

        # in scene_loader a robot doesn't have sensors
        if self.left_motor_controller is not None:
            self.left_motor_controller.sensor.draw(pos=pos, direction=self.direction)
            self.right_motor_controller.sensor.draw(pos=pos, direction=self.direction)

        # call super method to draw the robot
        super().draw(v, **kwargs)
