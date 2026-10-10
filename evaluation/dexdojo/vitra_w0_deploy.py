"""Sim-side episode drivers for the ``vitra_w0`` (WS bridge) policy.

This is the LingBot-VA variant of the stock vitra_w0 deploy: identical
protocol, but the single-env driver harvests keyframe observations at the
model's 4-actions-per-video-frame cadence and attaches them to the
chunk-boundary observation it hands to get_action (the stock model_client
swallows per-step update_obs calls client-side, so a bridge-side collector
would never see them). The LingBot bridge reads the extra keys in
update_obs and drives compute_kv_cache exactly like evaluation/robotwin.
"""

KEYFRAME_STRIDE = 4  # actions per sampled video frame (25 Hz sim, 6.25 fps video)


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    pending_keyframes = None
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        if pending_keyframes is not None:
            obs["lingbot_keyframes"] = pending_keyframes
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")

        keyframes = []
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if (action_idx + 1) % KEYFRAME_STRIDE == 0:
                keyframes.append(TASK_ENV.get_obs())
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
        pending_keyframes = keyframes or None


def eval_one_episode_batch(TASK_ENV, model_client):
    # LingBot-VA runs single-env lanes only (one AR stream per server);
    # the batch driver is intentionally unsupported here.
    raise NotImplementedError(
        "LingBot-VA lanes run with --eval_batch false / num_envs 1; "
        "the batch driver is not supported")
