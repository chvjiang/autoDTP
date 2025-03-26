from openai import OpenAI

import base64

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    image_format = "png" if image_path.lower().endswith(".png") else "jpeg"
    return f"data:image/{image_format};base64,{image_encoded}" 

client = OpenAI(
    api_key = "sk-gES1qlU8KXz6Y986eu1ib4iGzcXAFmI2nDgd2I6pHi3xSus3",
    base_url = "https://boyuerichdata.chatgptten.com/v1/",
)
content = [
    {
        "type": "text",
        "text": "Here are 3 texture maps of a hat, please evaluate their quality from the aspects of aesthetic, consistency, artifacts, and so on."
    },
    {
        "type": "image_url",
        "image_url": {"url": encode_image_to_base64("/mnt/petrelfs/chuziyang/FlashTex/0_hunyuan.png")}
    },
    {
        "type": "image_url",
        "image_url": {"url": encode_image_to_base64("/mnt/petrelfs/chuziyang/FlashTex/0_Trellis.png")}
    },
    {
        "type": "image_url",
        "image_url": {"url": encode_image_to_base64("/mnt/petrelfs/chuziyang/FlashTex/0_LN3Diff.png")}
    },
]
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": content,
        }
    ],
    model="gpt-4o",
)

print(chat_completion.choices[0].message.content)