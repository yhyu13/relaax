from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

import da3c_model


class DA3CParameterServer(parameter_server_base.ParameterServerBase):
    def close(self):
        self.session.close()

    def initialize_algorithm(self):
        self.session = session.Session(da3c_model.SharedParameters())
        self.session.op_initialize()

    def make_checkpoint(self):
        return self.session.make_checkpoint()

    def get_session(self):
        return self.session

    def n_step(self):
        return self.session.op_n_step()