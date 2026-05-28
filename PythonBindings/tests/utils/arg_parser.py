import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="LuisaCompute Python example")
	parser.add_argument(
		"--backend",
		type=str,
		default="metal",
		choices=["cuda", "dx", "metal", "vk", "fallback", "cpu", "remote"],
		help="Compute backend to use (default: metal)",
	)
	parser.add_argument(
		"--headless",
		action="store_true",
		help="Run without GUI",
	)
	parser.add_argument(
		"--advance_frames",
		type=int,
		default=30,
		help="Number of simulation frames to advance in headless mode (default: 30)",
	)
	return parser.parse_args()