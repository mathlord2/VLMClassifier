import re

class VQAEvaluator:
    def __init__(self):
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
            "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
            "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
            "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
            "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
            "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
            "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
            "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
            "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
            "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
            "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
            "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
            "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
            "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
            "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
            "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
            "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
            "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
            "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
            "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
            "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
            "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
            "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've"
        }
        self.manualMap = {
            'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
        }
        self.articles = ['a', 'an', 'the']
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']
    
    def extract_answer(self, text):
        """
        Extracts the answer from a string containing 'answer is: <answer>.' pattern.

        Args:
            text (str): Input string containing multiple sentences.
            
        Returns:
            boolean: Whether the pattern 'answer is: <answer>' was found or not
            str: The extracted answer or None if no match is found.
        """
        # Search for "answer is: " pattern in the text
        start_pattern = r'answer is: '
        
        # Find the starting position of the answer
        match = re.search(start_pattern, text)
        if not match:
            return False, (text.split('.')[-2] if '.' in text else text)
        
        start_pos = match.end()
        
        # Get the substring from the start position to the end
        answer = text[start_pos:]
        
        # String unnecessary characters like '.', '[', ']'
        if answer[-1] == '.':
            answer = answer[:-1]
        
        if answer[0] == '[':
            answer = answer[1:]
        
        if answer[-1] == ']':
            answer = answer[:-1]
        
        if answer is None:
            answer = ""
        
        return True, answer.strip()

    def preprocess(self, text):
        """
        Preprocess the input text to standardize it for comparison.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        text = text.lower()
        words = text.split()
        processed_words = []
        for word in words:
            if word in self.contractions:
                processed_words.append(self.contractions[word])
            else:
                processed_words.append(word)
        text = ' '.join(processed_words)
        
        words = text.split()
        processed_words = []
        for word in words:
            processed_words.append(self.manualMap.get(word, word))
        text = ' '.join(processed_words)
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.articles]
        text = ' '.join(filtered_words)
        
        text = self.commaStrip.sub(r'\1\3', text)
        text = self.periodStrip.sub('', text)
        
        for p in self.punct:
            text = text.replace(p, ' ')
        
        text = ' '.join(text.split())
        return text.strip()

    def evaluate(self, model_answer, human_answers):
        """
        Evaluate the model's answer against a set of human answers.

        Args:
            model_answer (str): The model's answer to evaluate.
            human_answers (list of str): A list of human answers to compare against.

        Returns:
            float: The accuracy score for the model's answer.
        """
        if not human_answers:
            return 0.0
        
        found_pattern, processed_model = self.extract_answer(self.preprocess(model_answer))
        processed_humans = [self.preprocess(ans) for ans in human_answers]

        if found_pattern:
            m = sum(1 for ans in processed_humans if ans == processed_model)
        else:
            m = sum(1 for ans in processed_humans if ans in processed_model)
        
        if len(human_answers) < 3:
            acc = min(m, 1) if m > 0 else 0.0
        else:
            acc = min(m / (len(human_answers) // 3), 1) if m > 0 else 0.0
        return acc

    def batch_evaluate(self, model_answers, human_answers_list):
        """
        Evaluate a batch of model answers against corresponding sets of human answers.

        Args:
            model_answers (list of str): A list of model answers to evaluate.
            human_answers_list (list of list of str): A list of lists, where each sublist contains human answers
                                                     corresponding to a model answer.

        Returns:
            float: The average accuracy score across all model answers.
        """
        total_acc = 0.0
        total_samples = len(model_answers)
        for model_ans, human_ans in zip(model_answers, human_answers_list):
            acc = self.evaluate(model_ans, human_ans)
            total_acc += acc
        return total_acc / total_samples if total_samples > 0 else 0.0