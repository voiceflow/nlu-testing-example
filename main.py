import abc
import time
import uuid
from dataclasses import dataclass

import requests
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


class NLUTester(abc.ABC):
    @abc.abstractmethod
    def __init__(self, data):
        pass

    @abc.abstractmethod
    def run_tests(self):
        pass

    @abc.abstractmethod
    def compare_results(self):
        pass

    @abc.abstractmethod
    def save_results(self):
        pass

    @abc.abstractmethod
    def visualize_data(self):
        pass


def format_entities(entities):
    f = {}
    for i in entities:
        if i == '':
            continue
        name, value = i.split(":")
        f[name] = value
    return f


def format_vf_entities(entities):
    entities_new = {}
    for k, v in entities.items():
        entities_new[k] = entities[k]["value"]
    return entities_new


class VoiceflowNLUTester(NLUTester):
    @dataclass
    class NLUResponseValues:
        utterance: str
        intent: str
        resolved_intent: str
        confidence: float
        entities: list = None
        resolved_entities: list = None
        next_step = None

    def __init__(self, data: dict[str, list[list[str]]], dm_key, version=None, test_cast_path=None):
        """
        Initialize the voiceflow
        Args:
            data: a dict of utter
            dm_key: key to access the dm api, project api key
            version: which version of the project to use
            test_case_path: path to test cases file
        Raises:
            NotImplemented for test_case_path
        """
        self.data = {}
        self.key = dm_key
        self.dm_key = dm_key
        self.version = version
        self.base_url = f"https://general-runtime.voiceflow.com/state/user/"
        self.entity_results = None
        self.utterance_results = None
        if not test_cast_path:
            self.data = data
        else:
            raise NotImplemented
        self.entity_index_mapper, self.intent_index_mapper = self.index_mapper_initializer()

    def index_mapper_initializer(self) -> tuple[dict[str,int], dict[str,int]]:
        """
        From self.data creates index mappers that give each intent and entity type a value
        Returns: entity_index_mapper,intent_index_mapper

        """
        intent_index_mapper = {'None': 0}
        entity_index_mapper = {'': 0}
        e_i = 1
        # iterate over all the intent classes, find the entities and increment index by 1
        for intent, v in self.data.items():
            for combo in v:
                _, entities = combo[0], format_entities(combo[1:])
                for k, v in entities.items():
                    if k != '' and not (k in entity_index_mapper.keys()):
                        entity_index_mapper[k] = e_i
                        e_i += 1

        for i, key in enumerate(self.data.keys()):
            intent_index_mapper[key] = i + 1
        return entity_index_mapper, intent_index_mapper

    def run_tests(self):
        results = []
        for intent, v in self.data.items():
            for combo in v:
                utterance, entities = combo[0], format_entities(combo[1:])
                resolved_intent, confidence, resolved_entities, _ = self.send_request(utterance, str(uuid.uuid4()))
                results.append(
                    self.NLUResponseValues(utterance, intent, resolved_intent, confidence, entities, resolved_entities))

        my_array = np.array(results)
        i: VoiceflowNLUTester.NLUResponseValues
        # create utterance results
        self.utterance_results = np.array(
            [(i.utterance,
              self.intent_index_mapper[i.intent], self.intent_index_mapper[i.resolved_intent],
              i.confidence) for i in
             my_array])

        entity_results = []
        # create entity results, one line per entity returned
        for i in my_array:
            for grouped in zip(i.entities.items(), i.resolved_entities.items()):
                entity_actual_name, entity_actual_value = grouped[0]
                entity_predicted_name, entity_predicted_value = grouped[1]
                entity_results.append([i.utterance, self.entity_index_mapper[entity_actual_name],
                                       self.entity_index_mapper[entity_predicted_name]])

        self.entity_results = np.array(entity_results)

    def send_request(self, utterance, user_id):
        """
        Sends a requests to voiceflow dm api
        Args:
            utterance: the utterance to access
            user_id: a guid

        Returns:

        """
        #makes the logs verbose
        url = self.base_url + f"{user_id}/interact?logs=true"

        body = {"action": {"type": "text", "payload": utterance}}
        start_time = time.time()
        response = requests.post(
            url,
            json=body,
            headers={"Authorization": self.key},
        )
        # response time
        print(time.time() - start_time)
        # Log the response
        r = response.json()
        m = r[0]["payload"]["message"]
        #entity reprompt mode
        if r[2]['type'] == 'entity-filling':
            short = r[2]['payload']['intent']['payload']
            confidence = short['confidence']
            resolved_intent = short['intent']['name']
            entities = short["entities"]
            next_step = m
        # regular mode
        else:
            confidence = m['confidence']
            resolved_intent = m['resolvedIntent']
            if resolved_intent != 'None':
                entities = m["entities"]
                entities = format_vf_entities(entities)
                next_step = r[3]["payload"]
            else:
                entities = {}
                next_step = None
        return resolved_intent, confidence, entities, next_step

    def compare_results(self, result_type=None):
        if result_type == 'utterance':
            acc = [self.utterance_results]
        elif result_type == 'entities':
            acc = [self.entity_results]
        else:
            acc = [self.entity_results, self.utterance_results]

        for results in acc:
            truth, pred = results[:, 1], results[:, 2]
            f1 = f1_score(truth, pred, average=None)
            print("f1 scores", f1)
            print("total f1", np.mean(f1))
            assert f1.all() > 0.85
            assert np.mean(f1) > 0.9

    def save_results(self):
        np.savetxt("entity_results.csv", self.entity_results, delimiter=",", fmt='%s')
        np.savetxt("utterance_results.csv", self.utterance_results, delimiter=",", fmt='%s')

    def visualize_data(self, result_type='utterance'):
        """

        Args:
            result_type:

        Returns:

        """
        if result_type == 'utterance':
            acc = [(self.utterance_results,self.intent_index_mapper.keys())]
        elif result_type == 'entities':
            acc = [(self.entity_results,self.entity_index_mapper.keys())]
        else:
            acc = [(self.utterance_results,self.intent_index_mapper.keys()), (self.entity_results,self.entity_index_mapper.keys())]

        for results,l in acc:

            np.seterr(divide='ignore', invalid='ignore')
            truth, pred = results[:, 1], results[:, 2]

            cm = confusion_matrix(truth, pred)
            import seaborn as sns
            import matplotlib.pyplot as plt
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            ax = sns.heatmap(cm, annot=True)
            ax.xaxis.set_ticklabels(list(l))
            ax.yaxis.set_ticklabels(list(l))
            plt.rcParams['savefig.transparent'] = True
            plt.show()


test_intents_1 = {
    "order_pizza": [("I'd like a large pizza", "size:large"), ("small cheese pizza", "size:small", "type:cheese"),
                    ("give me pizza", "")],
    "order_fries": [("poutine please", ""), ("do you have fries?", ""), ("can i get fries with mayo", "")],
    "help_me": [("Assist me!", ""), ("i need your help", ""), ("please help", "")],
}

vf_test = VoiceflowNLUTester(test_intents_1, dm_key="")
vf_test.run_tests()
vf_test.compare_results()
vf_test.save_results()
vf_test.visualize_data()
