from PIL import Image
import urllib.request


# Load an image from a URL and resize so that it is no bigger than 512x512
# and then save it to a file
def load_and_resize_image(path, filename):
    # Load the image from the URL
    image_file = urllib.request.urlopen(path)
    image = Image.open(image_file)

    print(f"loaded input image from {path}")

    w, h = image.size
    print(f"image size ({w}, {h})")

    old_size = image.size  # old_size[0] is in (width, height) format

    ratio = float(512) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    image.thumbnail(new_size, Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", (512, 512))
    new_image.paste(image, ((512 - new_size[0]) // 2,
                      (512 - new_size[1]) // 2))

    w, h = new_image.size
    print(f"Image resized to size ({w}, {h}) ")

    # Save the image to a file
    new_image.save(filename)


if __name__ == "__main__":
    # Load an image from a URL and save it to a file
    load_and_resize_image("https://cdn.photoswipe.com/photoswipe-demo-images/photos/home-demo/luca-bravo-ny6qxqv_m04-unsplash_snrzpf/luca-bravo-ny6qxqv_m04-unsplash_snrzpf_c_scale,w_1355.jpg", "test.png")

