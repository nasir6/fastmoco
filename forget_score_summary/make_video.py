import os
import moviepy.video.io.ImageSequenceClip
from glob import glob

# Source: https://bamidele01.medium.com/how-i-created-a-video-from-image-sequence-for-my-presentation-using-moviepy-in-python-e00f13d110e7

def images_to_video(image_folder_path: str, fps, extension:str, video_name:str, output_format:str):
    
    # images = [image_folder_path+'/'+img for img in os.listdir(image_folder_path) if img.endswith(extension)]
    images = glob(os.path.join(image_folder_path, f"*{extension}"))
    # images = sorted(images, key=lambda x: int(os.path.basename(x).split("upto")[1].rstrip(".png")))

    images = sorted(images, key=lambda x: int(os.path.basename(x).split("epoch_")[1].rstrip(".png")))
    
    movie_clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps)
    movie_clip.write_videofile(f"{video_name}{output_format}")

if __name__ == "__main__":
    # images_path = "/nfs/projects/healthcare/nasir/Fast-MoCo/forget_score_summary/figs"
    # images_to_video(images_path, 1, ".png", "forget_scores", ".mp4")

    images_path = "/nfs/projects/healthcare/nasir/Fast-MoCo/analyze_scores"
    images_to_video(images_path, 1, ".png", "cosine_scores", ".mp4")
