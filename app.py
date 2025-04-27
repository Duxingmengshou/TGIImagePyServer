import base64
import random

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from matplotlib import pyplot as plt
from utils.OpUtils import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_gamma_noise,
    add_random_erasing,
    cut_corner,
)
from utils.OpUtils import noisy_encrypt_decrypt

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def hello_world():  # put application's code here
    return "Hello World!"


@app.route("/check")
def check():  # put application's code here
    return jsonify({"status": "success", "message": "Hello World!", "data": None})


@app.route("/upload/image", methods=["POST"])
def process_image():
    try:
        # 获取JSON数据
        data = request.get_json()

        if not data or "data" not in data:
            return jsonify({"status": "error", "message": "Missing image data"}), 400

        # 提取Base64字符串
        image_data = data["data"]

        # 检查是否是有效的Base64图像数据
        if not image_data.startswith("data:image/"):
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        # 分离Base64头和数据部分
        header, encoded = image_data.split(",", 1)

        # 解码Base64数据
        decoded_image = base64.b64decode(encoded)

        # 将字节数据转换为numpy数组
        np_arr = np.frombuffer(decoded_image, np.uint8)

        # 使用OpenCV解码图像
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return (
                jsonify({"status": "error", "message": "Failed to decode image"}),
                400,
            )

        encrypted_img, perm_high_img, perm = noisy_encrypt_decrypt(
            img, key=123, mode="encrypt"
        )
        decrypted_img = noisy_encrypt_decrypt(
            (encrypted_img, perm_high_img, perm), key=123, mode="decrypt"
        )

        # ========== 将处理后的图像转换为Base64返回 ==========
        _, buffer = cv2.imencode(".png", encrypted_img)
        encrypted_base64 = base64.b64encode(buffer).decode("utf-8")

        _, buffer = cv2.imencode(".png", decrypted_img)
        decrypted_base64 = base64.b64encode(buffer).decode("utf-8")

        # 返回成功响应
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Image encrypted successfully",
                    "encrypted_data": f"data:image/png;base64,{encrypted_base64}",
                    "decrypted_data": f"data:image/png;base64,{decrypted_base64}",
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"Failed to process image: {str(e)}"}
            ),
            500,
        )


@app.route("/upload/encrypt_image", methods=["POST"])
def encrypt_image():
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"status": "error", "message": "Missing image data"}), 400

        image_data = data["data"]

        if not image_data.startswith("data:image/"):
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        header, encoded = image_data.split(",", 1)
        decoded_image = base64.b64decode(encoded)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return (
                jsonify({"status": "error", "message": "Failed to decode image"}),
                400,
            )

        encrypted_img, perm_high_img, perm = noisy_encrypt_decrypt(
            img, key=123, mode="encrypt"
        )

        # 编码加密后的图
        _, buffer_enc = cv2.imencode(".png", encrypted_img)
        encrypted_base64 = base64.b64encode(buffer_enc).decode("utf-8")

        # 保存perm_high_img和perm作为加密辅助信息
        perm_high_bytes = perm_high_img.tobytes()
        perm_high_base64 = base64.b64encode(perm_high_bytes).decode("utf-8")

        perm_bytes = np.array(perm, dtype=np.int32).tobytes()
        perm_base64 = base64.b64encode(perm_bytes).decode("utf-8")

        return (
            jsonify(
                {
                    "status": "success",
                    "encrypted_data": f"data:image/png;base64,{encrypted_base64}",
                    "encrypt_info": {
                        "perm_high": perm_high_base64,
                        "perm": perm_base64,
                    },
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/upload/decrypt_image", methods=["POST"])
def decrypt_image():
    try:
        data = request.get_json()

        if not data or "encrypted_data" not in data or "encrypt_info" not in data:
            return jsonify({"status": "error", "message": "Missing fields"}), 400

        # 解码加密图像
        encrypted_data = data["encrypted_data"]
        header, encoded_enc = encrypted_data.split(",", 1)
        decoded_enc = base64.b64decode(encoded_enc)
        np_arr_enc = np.frombuffer(decoded_enc, np.uint8)
        encrypted_img = cv2.imdecode(np_arr_enc, cv2.IMREAD_COLOR)

        if encrypted_img is None:
            return (
                jsonify(
                    {"status": "error", "message": "Failed to decode encrypted image"}
                ),
                400,
            )

        # 解码加密信息
        encrypt_info = data["encrypt_info"]

        perm_high_base64 = encrypt_info.get("perm_high")
        perm_base64 = encrypt_info.get("perm")

        if not perm_high_base64 or not perm_base64:
            return (
                jsonify({"status": "error", "message": "Incomplete encrypt_info"}),
                400,
            )

        perm_high_bytes = base64.b64decode(perm_high_base64)
        perm_high_img = np.frombuffer(perm_high_bytes, dtype=np.uint8)
        perm_high_img = perm_high_img.reshape(encrypted_img.shape)

        perm_bytes = base64.b64decode(perm_base64)
        perm = np.frombuffer(perm_bytes, dtype=np.int32)

        # 解密
        decrypted_img = noisy_encrypt_decrypt(
            (encrypted_img, perm_high_img, perm), key=123, mode="decrypt"
        )

        _, buffer_dec = cv2.imencode(".png", decrypted_img)
        decrypted_base64 = base64.b64encode(buffer_dec).decode("utf-8")

        return (
            jsonify(
                {
                    "status": "success",
                    "decrypted_data": f"data:image/png;base64,{decrypted_base64}",
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(port=9013, host="0.0.0.0", debug=True)
