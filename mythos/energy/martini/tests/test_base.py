import pytest
from mythos.energy.martini.base import MartiniEnergyConfiguration, MartiniTopology


class TestMartiniEnergyConfiguration:
    def test_parameter_loading(self):
        conf = MartiniEnergyConfiguration(param_1 = 1.0, **{"param-2": 2.0})
        assert conf["param_1"] == 1.0
        assert conf["param-2"] == 2.0

    def test_opt_params(self):
        conf = MartiniEnergyConfiguration(param_1 = 1.0, **{"param-2": 2.0})
        assert conf.opt_params == {"param_1": 1.0, "param-2": 2.0}

    def test_coupling_parameters(self):
        conf = MartiniEnergyConfiguration(
            couplings={"proxy_param": ["param_a", "param_b"]},
            proxy_param = 3.0,
            param_c = 4.0,
        )
        assert "proxy_param" in conf
        assert "proxy_param" not in conf.params
        assert conf["param_a"] == 3.0
        assert conf["param_b"] == 3.0
        assert conf["param_c"] == 4.0

        assert conf.opt_params == {"proxy_param": 3.0, "param_c": 4.0}

    def test_coupling_parameter_conflict(self):
        with pytest.raises(ValueError, match="more than one coupling"):
            MartiniEnergyConfiguration(
                couplings={"proxy_param": ["param_a", "param_b"], "other_proxy": ["param_b"]},
                proxy_param = 3.0,
                other_proxy = 4.0,
            )

    def test_keyerror_on_missing_param(self):
        conf = MartiniEnergyConfiguration(param_1 = 1.0)
        with pytest.raises(KeyError, match="Parameter 'param_2' not found"):
            _ = conf["param_2"]

    def test_init_params_returns_self(self):
        conf = MartiniEnergyConfiguration(param_1 = 1.0)
        assert conf.init_params() == conf

    def test_configuration_concatenate_with_config(self):
        conf1 = MartiniEnergyConfiguration(param_1 = 1.0)
        conf2 = MartiniEnergyConfiguration(param_2 = 2.0)
        combined = conf1 | conf2
        assert combined["param_1"] == 1.0
        assert combined["param_2"] == 2.0

    def test_configuration_concatenate_with_dict(self):
        conf = MartiniEnergyConfiguration(param_1 = 1.0)
        combined = conf | {"param_2": 2.0}
        assert combined["param_1"] == 1.0
        assert combined["param_2"] == 2.0

class TestMartiniTopology:
    def test_topology_from_tpr(self):
        top = MartiniTopology.from_tpr("data/test-data/martini/energy/m2/bond/test.tpr")
        n_atoms = 1280  # Known system for test, all DMPC lipids with 1280 atoms, 768 angles and 1158 bonded neighbors
        n_angles = 768
        n_bonds = 1152
        assert len(top.atom_names) == n_atoms
        assert len(top.atom_types) == n_atoms
        assert top.residue_names == ("DMPC",) * n_atoms
        assert top.angles.shape == (n_angles, 3)
        assert top.bonded_neighbors.shape == (n_bonds, 2)
        assert top.unbonded_neighbors.shape == (n_atoms * (n_atoms - 1) // 2 - n_bonds, 2)  # All pairs minus bonded
