from openai import OpenAI
import gradio as gr
import requests
from PIL import Image

client = OpenAI()

def openai_create(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role":"system","content":"you are a strategy expert."},
            {"role": "user", "content":prompt}
        ],
        temperature=0.9,
        max_tokens=1020,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    print(response.choices)

    return response.choices[0].message.content

def chatgpt_clone(input, history):
    history=history or []
    s = list(sum(history, ()))
    s.append(input + '\n')
    inp = ' \n'.join(s)
    output = openai_create(inp)
    history.append((input, output))

    return history, history

text_block = gr.Blocks()

with text_block:
    gr.Markdown("""<hi><center>My ChatbotGPT<center></h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder="Type:")
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone, inputs=[
        message, state], outputs=[chatbot, state])
    
def openai_create_img(prompt):
    response = client.images.generate(
        prompt = prompt,
        n=1,
        size="1024x1024"
    )

    image_url = response.data[0].url
    r = requests.get(image_url, stream=True)
    img = Image.open(r.raw)
    return img

img_block = gr.Blocks()

with img_block:
    gr.Markdown("""<h1><center>My DALL-E<center></h1>""")
    new_image = gr.Image()
    message = gr.Textbox(placeholder="Type:")
    submit = gr.Button("SEND")
    submit.click(openai_create_img, inputs=[message], outputs=[new_image])

def openai_var_img(im):
    img = Image.fromarray(im)
    img = img.resize((1024, 1024))
    img.save("img1.png", "PNG"),

    response = client.images.create_variation(
        image=open("img1.png", "rb"),
        n=1,
        size="1024x1024"
    )

    image_url = response.data[0].url
    r = requests.get(image_url, stream=True)
    img = Image.open(r.raw)

    return img

img_var_block = gr.Blocks()

with img_var_block:
    gr.Markdown("""<h1><center>DALL - E Image Variator<center</h1>""")
    with gr.Row():
        im = gr.Image()
        im_2 = gr.Image()

    submitt = gr.Button("SEND")
    submit.click(openai_var_img, inputs=[im], outputs=[im_2])

demo = gr.TabbedInterface([text_block, img_block, img_var_block], [
    "Kyle's Bot", "Kyle DALL-E", "Kyle DALL-E image variator"])
if __name__ == "__main__":
    demo.launch(share=True) 