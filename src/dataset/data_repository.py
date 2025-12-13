
def get_data(filename):
    with open(filename, "r") as f:
        text = f.read().lower()
    print(text)
    return text