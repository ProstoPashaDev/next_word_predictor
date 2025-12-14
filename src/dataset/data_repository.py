
def get_data(filename):
    f = open(filename, "r")
    x = f.readline().strip()
    text = ""
    while x != "":
        if x.split(" ")[0] == "<bot>":
            text += (x + " <eos>\n")
        else:
            text += (x + " \n")
        x = f.readline().strip()
    print(text)
    return text