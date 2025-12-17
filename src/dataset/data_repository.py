
def get_data(filename, num_rows=99999999):
    f = open(filename, "r", encoding="UTF-8")
    x = f.readline()
    text = ""
    i = 1
    while x != "":
        #if x == "\\":
            #x = f.readline().strip()
            #continue
        #if x.split(" ")[0] == "<bot>":
            #text += (x + " <eos>\n")
        #else:
            #text += ("<s> " + x + " ")
        text += x
        x = f.readline()
        i += 1
        if i >= num_rows:
            break
    #print(text)
    return text