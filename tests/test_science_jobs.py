import pytest
import run

def test_job_rejects_unsupported_nonhygroscopic_initializer(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='Invalid scientific'):
        run.split_to_nodes([0.],1,kappa=0.)
    assert not list(tmp_path.glob('main_node*.py'))
