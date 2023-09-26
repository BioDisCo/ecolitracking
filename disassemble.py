import cv2
import os

def extract_frames(video_path, output_directory):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Loop through frames and save them as images
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Construct the output file path
        output_file = os.path.join(output_directory, f"image{frame_count:06d}.png")

        # Save the frame as an image
        cv2.imwrite(output_file, frame)

        frame_count += 1

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "202308100450290408282 - Trim.mp4"  # Replace with the actual video file path
    output_directory = "images"  # Replace with the desired output directory

    extract_frames(video_path, output_directory)
