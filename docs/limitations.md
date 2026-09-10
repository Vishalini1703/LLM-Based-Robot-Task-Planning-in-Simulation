# Limitations and Threats to Validity

- The world contains a small declared set of objects, locations, and high-level skills. Results do not generalise automatically to open kitchens.
- Temperature zero reduces variability but does not make a hosted model service perfectly deterministic.
- Prompt wording and the provider's deployed model revision may affect later repetitions.
- The final 150-trial backend uses deterministic logical execution. Webots physical behaviour was confirmed on three representative scenarios, not repeated for every trial.
- `KitchenObjectNet` is trained entirely on synthetic frames from one adapted Webots kitchen. Its high held-out and protocol scores do not establish performance on photographs, different kitchens, unseen object designs, or physical cameras.
- Simulator recognition metadata creates dataset/evaluation ground truth only. Normal `find` decisions use the custom ONNX CNN, but the five objects remain visually simple and the domain is deliberately constrained.
- Navigation uses a static 2D kitchen map and clearance-aware A* routing. It does not perceive unexpected or moving obstacles, replan online, or provide collision-aware grasp synthesis.
- Expected state is declared in configuration. Incorrect declarations could make a logically consistent result physically misleading.
- The sample contains 30 commands. Confidence intervals are reported, but a larger and more varied catalogue would give stronger external validity.
- Simulator success is not evidence of physical-robot safety, real-world robustness, or regulatory compliance.
- The 25 CNN-backed manipulation episodes use fresh Webots processes and validate mapped static-obstacle clearance; they are not evidence of dynamic collision avoidance or safe physical grasping.
- Participant feedback remains pending institutional approval and cannot be claimed as completed.
