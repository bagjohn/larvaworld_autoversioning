from ...ipc import LarvaMessage, Client


class RemoteBrianModelInterface(object):

    # remote_dt: duration of remote simulation per step in ms
    def __init__(self, server_host='localhost', server_port=5795, remote_dt=100):
        self.server_host = server_host
        self.server_port = server_port
        self.t_sim = int(remote_dt)
        self.step_cache = {}


    def executeRemoteModelStep(self, model_instance_id, t_sim=self.t_sim, t_warmup=0, **kwargs):
        # t_sim: duration of remote model simulation in ms
        # warmup: duration of remote model warmup in ms
        if model_instance_id not in self.step_cache:
            self.step_cache[model_instance_id] = 0

        msg = LarvaMessage(self.sim_id, model_instance_id, self.step_cache[model_instance_id],
                           T=t_sim, warmup=t_warmup, **kwargs)
        # send model parameters to remote model server & wait for result response
        with Client((self.server_host, self.server_port)) as client:
            [response] = client.send([msg])  # this is a LarvaMessage object again
            self.step_cache[response.model_id] = response.step
            print("LarvaMessage response: {}".format(response))
            return response