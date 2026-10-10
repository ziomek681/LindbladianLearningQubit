"""Thin wrapper giving the design loop a physical-parameter decoder."""
from model import MeasurementTransformer, z_to_theta


class LindbladDecoder(MeasurementTransformer):
    def theta_from_features(self, x):
        return z_to_theta(self(x))
