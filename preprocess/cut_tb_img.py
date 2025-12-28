from PIL import Image

input_path = "./tb_imgs/L16_token_ori.png"
output_path = "./figures/L16_token_img.png"

patch = 256
rows_out = 3
cols_out = 6

# ===== 读图 =====
img = Image.open(input_path)

# ===== 先裁出一整行 =====
crop_one_row = img.crop((0, 0, 18 * patch, patch))

# ===== 拆成单个 patch =====
patches = []
for col in range(18):
    left = col * patch
    p = crop_one_row.crop((left, 0, left + patch, patch))
    patches.append(p)

# 现在 patches 长度 = 18

# ===== 新建目标画布 =====
final_img = Image.new("RGB", (cols_out * patch, rows_out * patch))

# ===== 重新排成 3 × 6 =====
idx = 0
for r in range(rows_out):
    for c in range(cols_out):
        x = c * patch
        y = r * patch
        final_img.paste(patches[idx], (x, y))
        idx += 1

final_img.save(output_path)
print(f"Saved to {output_path}")
