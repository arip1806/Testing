def rgb_fft(image):
    f_size = 25
    fft_images=[]
    fft_images_log = []
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2((image[:, :, i])))
        fft_images.append(rgb_fft)
        fft_images_log.append(np.log(abs(rgb_fft)))

    return fft_images, fft_images_log
def apply_mask(input_image, mask):
    _, mask_thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    mask_bool = mask_thresh.astype('bool')
    input_image[mask_bool] = 1

    return input_image


def apply_mask_all(list_images, list_mask):
    final_result = []

    for (i,mask) in zip(list_images, list_mask):
        result = apply_mask(i,mask)
        final_result.append(result)
    return final_result
def create_canvas_draw_instance(background_image, key, height, width):

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(background_image),
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        height = height,
        width = width,
        key=key,
    )

    return canvas_result
