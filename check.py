import cv2
import numpy as np

# 辅助数据（全局保存打乱顺序）
perm_key_storage = {}


import cv2
import numpy as np

# 全局保存打乱顺序
perm_key_storage = {}


def add_gaussian_noise(img, mean=0, std=5):
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def add_salt_pepper_noise(img, amount=0.002, salt_vs_pepper=0.5):
    noisy_img = img.copy()
    num_pixels = img.shape[0] * img.shape[1]

    # 盐（白点）
    num_salt = int(np.ceil(amount * num_pixels * salt_vs_pepper))
    coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
    noisy_img[coords[0], coords[1]] = 255

    # 椒（黑点）
    num_pepper = int(np.ceil(amount * num_pixels * (1.0 - salt_vs_pepper)))
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
    noisy_img[coords[0], coords[1]] = 0

    return noisy_img


def add_gamma_noise(img, shape=1.0, scale=2.0):
    noise = np.random.gamma(shape, scale, img.shape)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def add_random_erasing(img, erase_ratio=0.02):
    noisy_img = img.copy()
    h, w, c = img.shape
    num_pixels = h * w
    num_erase = int(num_pixels * erase_ratio)

    coords = np.random.randint(0, h, size=num_erase), np.random.randint(
        0, w, size=num_erase
    )
    noisy_img[coords[0], coords[1]] = 0  # 设置为黑色（0）

    return noisy_img

def cut_corner(img, cut_ratio=(0.1, 0.1)):
    h, w, c = img.shape
    cut_x = int(w * cut_ratio[0])
    cut_y = int(h * cut_ratio[1])

    noisy_img = img.copy()
    noisy_img[0:cut_y, 0:cut_x] = 0  # 把指定区域置0（黑色）

    return noisy_img


def noisy_encrypt_decrypt(img, key=123, mode="encrypt", transmission_noise_level=2):
    global perm_key_storage

    rng = np.random.default_rng(seed=key)

    if mode == "encrypt":
        h, w, c = img.shape
        flat_img = img.reshape(-1, c)
        num_pixels = flat_img.shape[0]

        # 打乱
        perm = rng.permutation(num_pixels)
        perm_key_storage[key] = perm

        shuffled = flat_img[perm]

        # 加随机噪声
        noise = rng.integers(0, 256, size=shuffled.shape, dtype=np.uint8)
        encrypted = (shuffled.astype(np.int16) + noise.astype(np.int16)) % 256
        encrypted = encrypted.astype(np.uint8)

        encrypted_img = encrypted.reshape(h, w, c)
        perm_high_img = noise.reshape(h, w, c)

        # --- 模拟传输噪声 ---
        if transmission_noise_level > 0:
            # 小的均匀噪声
            trans_noise_img = rng.integers(
                -transmission_noise_level,
                transmission_noise_level + 1,
                size=encrypted_img.shape,
            )
            trans_noise_key = rng.integers(
                -transmission_noise_level,
                transmission_noise_level + 1,
                size=perm_high_img.shape,
            )

            encrypted_img = (encrypted_img.astype(np.int16) + trans_noise_img) % 256
            encrypted_img = encrypted_img.astype(np.uint8)

            perm_high_img = (perm_high_img.astype(np.int16) + trans_noise_key) % 256
            perm_high_img = perm_high_img.astype(np.uint8)

            # 再叠加更多真实噪声（高斯噪声、椒盐噪声、伽马噪声）
            # encrypted_img = add_gaussian_noise(encrypted_img, std=20)
            # encrypted_img = add_salt_pepper_noise(encrypted_img, amount=0.002)
            # encrypted_img = add_gamma_noise(encrypted_img, shape=1.0, scale=2.0)
            # encrypted_img = add_random_erasing(encrypted_img, erase_ratio=0.25)
            encrypted_img = cut_corner(encrypted_img, cut_ratio=(0.95, 0.95))
            # encrypted_img = add_salt_pepper_noise(encrypted_img, amount=0.002)
            # encrypted_img = add_gamma_noise(encrypted_img, shape=1.0, scale=2.0)

            # perm_high_img = add_gaussian_noise(perm_high_img, std=30)
            # perm_high_img = add_salt_pepper_noise(perm_high_img, amount=0.002)
            # perm_high_img = add_gamma_noise(perm_high_img, shape=1.0, scale=2.0)

        return encrypted_img, perm_high_img

    elif mode == "decrypt":
        if isinstance(img, tuple) and len(img) == 2:
            encrypted_img, perm_high_img = img
            h, w, c = encrypted_img.shape

            encrypted_flat = encrypted_img.reshape(-1, c)
            noise_flat = perm_high_img.reshape(-1, c)

            perm = perm_key_storage.get(key)
            if perm is None:
                raise ValueError("No permutation key found for the given key.")

            # 注意这里也需要astype防止负数
            shuffled = (
                encrypted_flat.astype(np.int16) - noise_flat.astype(np.int16)
            ) % 256
            shuffled = shuffled.astype(np.uint8)

            inv_perm = np.argsort(perm)
            recovered = shuffled[inv_perm]

            decrypted_img = recovered.reshape(h, w, c)
            return decrypted_img
        else:
            raise ValueError(
                "For decryption, img must be a tuple (encrypted_img, perm_high_img)."
            )
    else:
        raise ValueError("Mode must be either 'encrypt' or 'decrypt'.")


# 使用示例
if __name__ == "__main__":
    img = cv2.imread("./oo.jpg")
    cv2.imshow("Original", img)

    encrypted_img, perm_high_img = noisy_encrypt_decrypt(img, key=123, mode="encrypt")
    cv2.imshow("Encrypted", encrypted_img)

    decrypted_img = noisy_encrypt_decrypt(
        (encrypted_img, perm_high_img), key=123, mode="decrypt"
    )
    cv2.imshow("Decrypted", decrypted_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
