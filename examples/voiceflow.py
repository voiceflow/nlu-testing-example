from src.main import VoiceflowNLUTester


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
