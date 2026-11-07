from sa_jev.smoke import run


def test_train_reference_dpo_checkpoint_eval_and_figures(tmp_path):
    summary=run(tmp_path)
    assert summary["checkpoint_roundtrip"]
    assert summary["stage1_optimizer_steps"]==2 and summary["stage2_optimizer_steps"]==2
    assert summary["evaluation_errors"]==0
    for f in ("stage1/training.csv","stage2/training.pdf","evaluation/results.csv","evaluation/figure.png"):
        assert (tmp_path/f).is_file()
