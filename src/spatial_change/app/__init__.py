"""The local web app: two videos in, the changes and the scene out.

``handlers`` holds the steps behind the app and never imports gradio; ``ui``
builds the page and needs the ``app`` extra. Importing the package does not
load gradio, torch, transformers, open3d, or cv2.
"""
