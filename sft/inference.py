from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    model_name_or_path = ""
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device = "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    input_string = ""
    message = [
        {'role': 'user', 'content':input_string}
    ]
    output_text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    input_ids = tokenizer(input, return_tensors='pt').to(model.device)
    output = model.generate(**input_ids)
    print(tokenizer.decode(output[0][len(input_ids.input_ids[0]):]))