f = open("C:/KhramovPavel/Project/Python/NextWordPredictor/recources/dial_eval.txt", encoding="UTF-8")

x = f.readline().strip().split("<eos>")
text = ""
j = 0
while x != [""]:
    textx = ""
    for i in range(len(x)):
        if x[i] != "" and x[i][0] == " ":
            textx += x[i][1::] + "<eos>\n"
        elif x[i] != "":
            textx += x[i] + "<eos>\n"
    text += textx
    print(j)
    x = f.readline().strip().split("<eos>")
    j += 1

print(text)

f2 = open("/recources/dialog_eval_string_split.txt", "w", encoding="UTF-8")
f2.write(text)