# 今晚训练清单 · 庄方宜第二代（GPT-SoVITS v5 full-FT）

## 入口

ModelScope 训练包：

https://modelscope.cn/datasets/yangwenjie1231/zfy-v5-fullft-pkg

打开根目录的：

```text
zfy_v5_fullft_final.ipynb
```

## 运行顺序

1. 定位/克隆训练包
2. 拉取最终流水线脚本
3. 安装 SoVITS 训练依赖（不重装 torch）
4. `doctor`
5. `prepare`
6. 确认输出里出现：
   ```text
   推荐模式: quantizer 仍冻结
   CBM_DELTA=0.00000000
   ZFY_FULL_EXPORT_ALL_PATCH
   ```
7. `launch`
8. 用 `status` 监控
9. 出现 e8/e16/e24/e32 后执行 `validate`
10. 合成 4 个 epoch 的验收音频
11. 无提前终止后执行 `upload`

## 正式训练参数

```text
batch=16
epochs=32
save_every_epoch=8
CUDA_VISIBLE_DEVICES=0
ZFY_UNFREEZE=enc_p,ssl_proj
ZFY_KEEP_CKPTS=2
quantizer=冻结且导出
```

## 输出

```text
SoVITS_weights_v5_full_e32_final/
  v5_full_e32_final_e8_s432_l0.pth
  v5_full_e32_final_e16_s864_l0.pth
  v5_full_e32_final_e24_s1296_l0.pth
  v5_full_e32_final_e32_s1728_l0.pth
```

实际 step 若因分桶变化略有不同，以文件名为准。

## 上传位置

正式模型仓库：

https://modelscope.cn/models/yangwenjie1231/zhuang-fangyi-voice-v2

已预传：
- GPT `zfh-e4.ckpt`
- 默认参考音频
- normal / firm / gentle 三种风格
- 模型卡

训练完成后流水线上传：
- 正式 e32 到 `torch_weights/`
- e8/e16/e24 到 `checkpoints/v5_fullft_final/`
- manifest / 配置 / 日志

## 红线

- 不要设置 `ZFY_UNFREEZE=all` 或 `1`
- 日志里不能出现 `quantizer` 在 unfrozen 列表
- 不要把旧 `nofreeze` 权重上传为正式模型
- `validate` 未通过前不要创建 GitHub Release
