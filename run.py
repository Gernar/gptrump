import torch
from transformers import AutoTokenizer


def remove_zeroes(inputs):
  i = inputs.shape[-1] - 1
  while inputs[0][i] == 0 and i > -1:
    i -= 1
  
  return inputs[:, :i+1]


model = torch.load('./gptrum-small.model')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')

model.to('cpu')
model.eval()

step = 0

while True:
    msg = input(">> User:")
    if msg == 'stop':
        break
    new_user_input_ids = tokenizer.encode(msg + tokenizer.eos_token, return_tensors='pt')

    # bot_input_ids = torch.cat([remove_zeroes(chat_history_ids), new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    bot_input_ids = new_user_input_ids

    bot_input_ids = bot_input_ids[:, -50:]

    chat_history_ids = model.generate(
        bot_input_ids, max_length=100,
        pad_token_id=tokenizer.eos_token_id,  
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature = 0.9
    )

    step += 1

    print("TrumpBot: {}".format(tokenizer.decode(remove_zeroes(chat_history_ids[:, bot_input_ids.shape[-1]:])[0], skip_special_tokens=True)))