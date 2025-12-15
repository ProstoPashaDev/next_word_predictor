f = open("C:/KhramovPavel/Project/Python/NextWordPredictor/recources/dialogues_validation.txt", encoding="UTF-8")

x = f.readline().strip().split("__eou__")
text = ""
j = 0
while x != [""]:
    text += "<s> "
    textx = ""
    for i in range(len(x)):
        if x[i] == "":
            continue
        if i % 2 == 0:
            textx += "<user> " + x[i]
        else:
            textx += "<bot>" + x[i] + "<eos> "
    text += (textx + "\n")
    print(j)
    x = f.readline().strip().split("__eou__")
    j += 1


print(text)

f2 = open("/recources/dial_eval.txt", "w", encoding="UTF-8")
f2.write(text)