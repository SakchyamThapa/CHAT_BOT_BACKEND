def clean_final_ans(answer: str) -> str:
    all_text = ""
    for match in answer.matches:
        all_text += match['metadata']['text'] + " "
    return all_text
