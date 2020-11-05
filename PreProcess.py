import datefinder
import nltk
import spacy
import en_core_web_sm
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer

count = 1

def tokenization(text):
    # Various different patterns to help the program tokenize the document
    # (?:[A-Za-z]\.)+ allows abbreviations such as U.S.A.
    # (?:[A-Za-z]\'\w+)+ allows name that contains apostrope
    # \$?\d+\b(?:\.\d+|\,\d+)?%? allows number normalizer
    # (?:Mr\.\s*\w+|Mrs\.\s*\w+|Mrs\.\s*\w+|Ms\.\s*\w+|Miss\.?\s*\w+) allows titles such as Mr. John, Mrs. Smith...
    # \'s allows possessive such as U.S. Embassy's annual agriculture report
    # \w+(?:-\w+)* allows words with optional hyphen between two words
    # [/.,?!;"'():_-`] some special tokens such as comma, dot, question mark...
    # \.\.\. allows ellipsis
    pattern = r"""(?x) 
          (?:[A-Za-z]\.)+
        | (?:[A-Za-z]\'\w+)+
        | \$?\d+\b(?:\.\d+|\,\d+)?%?
        | (?:Mr\.\s*\w+|Mrs\.\s*\w+|Mrs\.\s*\w+|Ms\.\s*\w+|Miss\.?\s*\w+)
        | \'s
        | \w+(?:-\w+)*
        | [/.,?!;"'():_-`]
        | \.\.\.
    """

    # We will use a regular express based tokenizer
    tokenizer = RegexpTokenizer(pattern)

    tokens = tokenizer.tokenize(text)
    return tokens


def sentSplitting(text):
    #(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s
    # We use a normal sentence tokenizer provided by NLTK
    sentences = nltk.sent_tokenize(text)
    for i in range(len(sentences)):
        # Since the document contains newline characters
        # we will remove all of them
        temp = re.sub(r"\n","", sentences[i])
        sentences[i] = temp
    return sentences

def posTagging(tokens):
    # we will convert all the tokens to lowercase
    # We use the provided pos_tag() function in NLTK
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    POS = nltk.pos_tag(tokens)
    return POS


def MeasuredEntityDetection(POS):
    # We use the PorterStemmer to obtain the original word of the unit measurement
    stemmer = PorterStemmer()

    # A text file that contains different units of measurement
    filename = "measurement.txt"

    with open(filename) as f:
        content = f.read().split(", ")

    # We will convert all the units into lowercase
    content = [s.lower() for s in content]

    result = set()
    for i in range(len(POS)):
        # We will find the token that has been tagged as CD
        if POS[i][1] == "CD":
            found = False
            string = POS[i][0]
            # The measurement unit usually is right next to the number, so we only need to check for the next token
            # if there is any
            if i + 2 < len(POS):
                for j in range(i + 1, i + 2):
                    measureUnit = ""
                    # We retrieve the original unit by using stemmer
                    word = stemmer.stem(POS[j][0])
                    if "-" in word:
                        temp = word.split("-")
                        for w in temp:
                            measureUnit = stemmer.stem(w)
                            if measureUnit.lower() in content or measureUnit.lower() == "tonn" or measureUnit.lower() == "ct":
                                string += " " + POS[j][0]
                                found = True
                                break;
                    elif word.lower() in content or word.lower() == "tonn" or word.lower() == "ct":
                        string += " " + POS[j][0]  # We have a special case for tonnes and cts
                        found = True  # the original of "tonnes" is "tonne", but the stemmer reduce "tonnes" to "tonn" instead
                        break;  # As for cts, the stemmer will reduce it to ct
                    elif found is not True:  # Hence, we add another corner case in the if condition
                        string += " " + POS[j][0]
                if found == True:
                    result.add(string)

    return result


def DateRecognizerCFG(text, POS):
    # A context free grammar that we will use to detect dates in the text
    dateGrammar = nltk.CFG.fromstring("""
        DATE -> YEAR | MONTH SEP DAY SEP YEAR | YEAR SEP MONTH SEP DAY | WORDMONTH | WORDDAY | WORDMONTH YEAR
        DATE -> ORD WORDMONTH YEAR | ORD WORDMONTH | DAY WORDMONTH | WORDMONTH DAY | WORDMONTH DAY YEAR | WORDMONTH ORD | WORDMONTH ORD YEAR
        DATE -> DET ORD DATEAUX WORDMONTH YEAR | DET ORD DATEAUX WORDMONTH | ORD DATEAUX WORDMONTH YEAR | ORD DATEAUX WORDMONTH
        DATE -> WORDDAY WORDMONTH DAY YEAR
        SEP -> "/" | "-" | "."
        YEAR -> NUM NUM NUM NUM
        MONTH -> NUM | NUM NUM 
        WORDMONTH -> "january" | "february" | "march" | "april" | "may" | "june" | "july"
        WORDMONTH -> "august" | "september" | "october" | "november" | "december"
        DAY -> NUM | NUM NUM
        NUM -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        ORD -> "1st" | "2nd" | "3rd" | "4th" | "5th" | "6th" | "7th" | "8th" | "9th" | "10th" | "11th" | "12th"
        ORD -> "13th" | "14th" | "15th" | "16th" | "17th" | "18th" | "19th" | "20th" | "21st" | "22nd" | "23rd"
        ORD -> "24th" | "25th" | "26th" | "27th" | "28th" | "29th" | "30th" | "31st"
        ORD -> "first" | "second" | "third" | "fourth" | "fifth" | "sixth" | "seventh" | "eighth" | "ninth" | "tenth"
        ORD -> "eleventh" | "twelfth" | "thirteenth" | "fourteenth" | "fifteenth" | "sixteenth" | "seventeenth" | "eighteenth"
        ORD -> "nineteenth" | "twentieth" | "twenty-first" | "twenty-second" | "twenty-third" | "twenty-fourth"
        ORD -> "twenty-fifth" | "twenty-sixth" | "twenty-seventh" | "twenty-eighth" | "twenty-ninth" | "thirtieth" | "thirty-first"
        WORDDAY -> "monday" | "tuesday" | "wednesday" | "thursday" | "friday" | "saturday" | "sunday"
        DATEAUX -> "of"
        DET -> "the"
    """)

    # We use ChartParser provided by NLTK to generate the parse tree
    parser = nltk.ChartParser(dateGrammar)
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "21st", "22nd", "23rd",
                "24th", "25th", "26th", "27th", "28th", "29th", "30th", "31st",
                "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                "eighteenth",
                "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third", "twenty-fourth",
                "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth", "thirtieth",
                "thirty-first"]

    result = set()

    matches = datefinder.find_dates(text, source=True)

    matchesList = []

    for match in matches:
        matchesList.append(match[1])

    #print(matchesList)

    prepositions = [token[0].lower() for token in POS if token[1] == "IN"]
    prepositions = set(prepositions)

    for pre in prepositions:
        for i in range(len(matchesList)):
            found = False
            for ordinal in ordinals:
                if (pre in matchesList[i]) and (ordinal not in matchesList[i]) and (found is False):
                    matchesList[i] = matchesList[i].lower().replace(pre, "", 1).strip()
                    found = True

    for token in POS:
        for match in matchesList:
            if (token[0].lower() in months or token[0].lower() in days) and token[0] in match and len(token[0]) == len(
                    match):
                result.add(token[0])

    for match in matchesList:
        count = 0
        try:
            if len(match) == 2 and match.isnumeric() and int(match) <= 31:
                temp = [d for d in match]
                trees = parser.parse(temp)
                for tree in trees:
                    count += 1

                if count > 0:
                    result.add(match)
            elif len(match) == 4 and match.isnumeric() and int(match) <= 2020:
                temp = [d for d in match]
                trees = parser.parse(temp)
                for tree in trees:
                    count += 1

                if count > 0:
                    result.add(match)
            else:
                toks = []
                temp = []
                for tok in match.split():
                    #print(tok)
                    if "," in tok:
                        t = tok.replace(",", "").strip()
                        temp.append(t.lower())
                    else:
                        temp.append(tok.lower())

                for word in temp:
                    if word.isnumeric() and int(word) <= 2020:
                        for d in word:
                            toks.append(d)
                    else:
                        toks.append(word)
                trees = parser.parse(toks)
                for tree in trees:
                    # print(tree)
                    count += 1

                if count > 0:
                    result.add(match)
        except:
            # print("exception")
            continue

    return result


def DateParseCFG(date):
    # A context free grammar to parse a date string passed to this function
    dateGrammarParser = nltk.CFG.fromstring("""
        DATE -> YEAR | MONTH SEP DAY SEP YEAR | YEAR SEP MONTH SEP DAY | WORDMONTH | WORDDAY | WORDMONTH YEAR
        DATE -> ORD WORDMONTH YEAR | ORD WORDMONTH | DAY WORDMONTH | WORDMONTH DAY | WORDMONTH DAY YEAR | WORDMONTH ORD | WORDMONTH ORD YEAR
        DATE -> DET ORD DATEAUX WORDMONTH YEAR | DET ORD DATEAUX WORDMONTH | ORD DATEAUX WORDMONTH YEAR | ORD DATEAUX WORDMONTH
        DATE -> WORDDAY WORDMONTH DAY YEAR
        SEP -> "/" | "-" | "."
        YEAR -> NUM NUM NUM NUM | NUM NUM
        MONTH -> NUM | NUM NUM 
        WORDMONTH -> "january" | "february" | "march" | "april" | "may" | "june" | "july"
        WORDMONTH -> "august" | "september" | "october" | "november" | "december"
        DAY -> NUM | NUM NUM
        NUM -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        ORD -> "1st" | "2nd" | "3rd" | "4th" | "5th" | "6th" | "7th" | "8th" | "9th" | "10th" | "11th" | "12th"
        ORD -> "13th" | "14th" | "15th" | "16th" | "17th" | "18th" | "19th" | "20th" | "21st" | "22nd" | "23rd"
        ORD -> "24th" | "25th" | "26th" | "27th" | "28th" | "29th" | "30th" | "31st"
        ORD -> "first" | "second" | "third" | "fourth" | "fifth" | "sixth" | "seventh" | "eighth" | "ninth" | "tenth"
        ORD -> "eleventh" | "twelfth" | "thirteenth" | "fourteenth" | "fifteenth" | "sixteenth" | "seventeenth" | "eighteenth"
        ORD -> "nineteenth" | "twentieth" | "twenty-first" | "twenty-second" | "twenty-third" | "twenty-fourth"
        ORD -> "twenty-fifth" | "twenty-sixth" | "twenty-seventh" | "twenty-eighth" | "twenty-ninth" | "thirtieth" | "thirty-first"
        WORDDAY -> "monday" | "tuesday" | "wednesday" | "thursday" | "friday" | "saturday" | "sunday"
        DATEAUX -> "of"
        DET -> "the"
    """)

    # We also use the ChartParser in NLTK
    parser = nltk.ChartParser(dateGrammarParser)

    # We will need to tokenize the string and convert it into lowercase
    temp = tokenization(date)
    temp = [w.lower() for w in temp]
    nums = []
    toks = []

    #     for token in toks:
    #         if token.isnumeric():
    #             nums.append(token)

    # As the grammar rules require to split the number into individual character
    # we will need to check if the current token is a number or not and split it
    for word in temp:
        if word.isnumeric() and int(word) <= 2020:
            for d in word:
                toks.append(d)
        else:
            toks.append(word)

    # Lastly, we remove the comma if the list may contain it
    if "," in toks:
        toks.remove(",")

    #print(toks)
    try:
        count = 0;
        trees = parser.parse(toks)
        for tree in trees:
            print(tree)
            count += 1

        if count == 0:
            print("Invalid date format")
    except:
        print("Invalid date format")

def parseSentence(sent):
    global count
    temp = tokenization(sent)
    tokens = []
    for token in temp:
        if token != "," and token != ".":
            tokens.append(token)
    #We read the grammar from the text file
    grammar = """"""
    with open("grammar.txt") as file:
        grammar = file.read()
    CFG = nltk.CFG.fromstring(grammar)

    #We use Earley parsing algorithm
    parser = nltk.EarleyChartParser(CFG)
    trees = parser.parse(tokens)
    lst = [tree for tree in trees]
    return lst

def namedEntities(text):
    nlp = spacy.load("en_core_web_sm")
    entities = nlp(text)
    return entities

def PreProcess(filename="Data.txt", save=True):
    with open(filename) as file:
        for line in file:
            print("Data: ", line)
            tokens = tokenization(line)
            sents = sentSplitting(line)
            POS = posTagging(tokens)
            measuredEntities = MeasuredEntityDetection(POS)
            dates = DateRecognizerCFG(line, POS)
            entities = namedEntities(line)
            #If the option to save to file is False
            #We simply print the result to the screen
            if save is False:
                print("List of tokens: ")
                print(tokens)
                print("\nList of sentences:")
                for sent in sents:
                    print(sent)
                print("\nPOS: ")
                print(POS)
                if len(measuredEntities) > 0:
                    print("\nList of entities: ")
                    print(measuredEntities)
                else:
                    print("\nNo measure unit in the text")

                if len(dates) > 0:
                    print("\nDates recoginized: \n")
                    print(dates)
                else:
                    print("\nNo dates in text\n")

                print("\nList of entities: \n")
                for entity in entities.ents:
                    print(entity.text, entity.label_)

                for sent in sents:
                    trees = parseSentence(sent)
                    print("\nParse tree for sentence: " + sent + "\n")
                    print("\nNumber of parse trees: " + str(len(trees)))
                    for tree in trees:
                        print(tree, "\n")
                        print("---------------------------------------\n")
                print("---------------------------------------")
            #Else, we save the result to a text file
            else:
                global count
                name = filename[:filename.index(".")]
                with open("Result" + name + str(count) + ".txt", "w") as f:
                    f.write("Data: " + line + "\n")
                    f.write("\nList of tokens: ")
                    f.write(str(tokens))
                    f.write("\nList of sentences:")
                    for sent in sents:
                        f.write(str(sent))
                    f.write("\nPOS: ")
                    f.write(str(POS))
                    if len(measuredEntities) > 0:
                        f.write("\nList of entities: ")
                        f.write(str(measuredEntities))
                    else:
                        f.write("\nNo measure unit in the text")

                    if len(dates) > 0:
                        f.write("\nDates recoginized: \n")
                        f.write(str(dates))
                    else:
                        f.write("\nNo dates in text\n")

                    f.write("\nList of entities: \n")
                    for entity in entities.ents:
                        f.write(entity.text + " " + entity.label_ + "\n")

                    for sent in sents:
                        trees = parseSentence(sent)
                        f.write("\nParse tree for sentence: " + str(sent) + "\n")
                        f.write("\nNumber of parse trees: " + str(len(trees)) + "\n")
                        for tree in trees:
                            f.write(str(tree) + "\n")
                            f.write("---------------------------------------\n")
                    f.write("---------------------------------------")
                    count += 1
            
PreProcess(filename="Challenge.txt", save=True)