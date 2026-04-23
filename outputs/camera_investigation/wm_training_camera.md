# WM training camera — evidence

## Direct evidence from the checkpoint's hydra config

[outputs/pusht_cam1/.hydra/config.yaml](../../outputs/pusht_cam1/.hydra/config.yaml) lines 70-71:

```yaml
  obs_keys:
  - camera_1_color
```

This is the dataset config that was active when the WM was trained. `obs_keys` is the list
of HDF5 image keys consumed by the encoder. **Only `camera_1_color` is in the list** —
the WM's encoder never saw `camera_0_color`.

## Corroborating evidence

The checkpoint directory name itself: `outputs/pusht_cam1/checkpoints/best.ckpt` — the
suffix `_cam1` was written by the training launch script and matches the obs_keys above.

`obs_keys` is then propagated through the rest of the WM config:
- line 108: `obs_keys: ${dataset.obs_keys}` (passed to dynamics module)
- line 219: `obs_keys: ${dataset.obs_keys}` (passed to cost-fn cfg)
- line 273: `obs_keys: ${dataset.obs_keys}` (passed to autoencoder)

So every part of the WM (encoder, decoder, dynamics, autoencoder) is built around
`camera_1_color` as the single image stream.

## Conclusion

The WM is a **single-camera model bound to `camera_1_color`**.
