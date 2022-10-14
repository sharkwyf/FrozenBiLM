import json
from petrel_client.client import Client

conf_path = '~/petreloss.conf'
client = Client(conf_path)
img_url = 's3://ego4d/v1/full_scale/fffbaeef-577f-45f0-baa9-f10cabf62dfb.mp4'

img_bytes = client.get(img_url)
print(img_bytes)