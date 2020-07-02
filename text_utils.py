
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def DrawRegion(text, font, fontcolor, shadowcolor, shadow_radius = 1, spread = 0) :
	# we first get size of region we need to draw text
	text_width, text_height = font.getsize(text)
	text_height += 2 * shadow_radius + 2 * spread
	text_width += 2 * shadow_radius + 2 * spread

	# now we can create our outputs
	ret_text = Image.new("RGB", (text_width, text_height), (0, 0, 0))
	ret_mask = Image.new("L", (text_width, text_height), 0)

	# create draw objects
	draw_text = ImageDraw.Draw(ret_text)
	draw_mask = ImageDraw.Draw(ret_mask)

	# draw text
	draw_text.text((spread - shadow_radius, spread - shadow_radius), text, font=font, fill=shadowcolor, stroke_width = shadow_radius)
	draw_text.text((spread, spread), text, font=font, fill=fontcolor)
	# draw mask
	draw_mask.text((spread - shadow_radius, spread - shadow_radius), text, font=font, fill=255, stroke_width = shadow_radius)
	draw_mask.text((spread, spread), text, font=font, fill=255)

	return np.array(ret_text), np.array(ret_mask)



