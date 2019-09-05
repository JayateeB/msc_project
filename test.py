from tasks.sim_graph_data_generator import SimGraphDataGenerator

generator = SimGraphDataGenerator()
generator.generate("givenchy_simulation_result_6hrs_6_hrs_model_retrained.csv", "6hrs_sim_graph_retrained.json", 360.0)