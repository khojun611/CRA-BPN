import os
import cv2
import numpy as np
import argparse

def load_points(file_path):
    """
    텍스트 파일에서 여러 개의 (x, y) 포인트로 이루어진 폴리곤들을 불러옵니다.
    각 폴리곤은 줄바꿈으로 구분됩니다.
    """
    polygons = []
    with open(file_path, 'r') as f:
        for line in f:
            points = list(map(int, line.strip().split(',')))
            points = np.array(points).reshape(-1, 2)
            polygons.append(points)
    return polygons

def draw_polygons(image, polygons):
    """
    이미지에 여러 개의 폴리곤을 그립니다.
    """
    for points in polygons:
        polygon = points.reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

def main(image_dir, points_dir, output_dir):
    """
    이미지를 불러오고 여러 개의 폴리곤을 그린 후 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        points_path = os.path.join(points_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        output_path = os.path.join(output_dir, image_file)

        # 이미지 불러오기
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image from {image_path}")
            continue

        # 포인트 불러오기
        polygons = load_points(points_path)
        if not all(len(points) == 14 for points in polygons):
            print(f"Error: Points file {points_path} does not contain 14 (x, y) pairs for some polygons")
            continue

        # 폴리곤 그리기
        image_with_polygons = draw_polygons(image, polygons)

        # 이미지 저장
        cv2.imwrite(output_path, image_with_polygons)
        print(f"Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw polygons on images and save the results.')
    parser.add_argument('--image_dir', required=True, help='Directory containing images.')

    args = parser.parse_args()

    points_dir = './inference'
    output_dir = './inference_vis'

    main(args.image_dir, points_dir, output_dir)
