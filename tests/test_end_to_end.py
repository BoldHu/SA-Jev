from pathlib import Path
from sa_jev.smoke import run_fresh


def test_train_reference_dpo_checkpoint_eval_and_figures(tmp_path):
    prior = {}
    for _ in range(2):
        summary = run_fresh(tmp_path)
        assert summary["checkpoint_roundtrip"]
        assert summary["stage1_optimizer_steps"] == 2 and summary["stage2_optimizer_steps"] == 2
        assert summary["evaluation_errors"] == 0
        output = Path(summary["output_directory"])
        assert output.parent == tmp_path / "smoke"
        for f in ("stage1/training.csv", "stage2/training.pdf", "evaluation/results.csv", "evaluation/figure.png"):
            assert (output / f).is_file()
        for path, content in prior.items():
            assert path.read_bytes() == content
        prior[output / "verification.json"] = (output / "verification.json").read_bytes()
        checkpoint = output / "stage1/step_000002/trainable.safetensors"
        prior[checkpoint] = checkpoint.read_bytes()
    assert len(list((tmp_path / "smoke").iterdir())) == 2
