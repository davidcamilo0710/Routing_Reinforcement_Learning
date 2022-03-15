# Routing Elephants Reinforcement Learning

Internet has become a necessity for all people and every day the traffic grows even more. Intelligently routing traffic is important to avoid network congestion and maintain high quality standards (speed and bandwidth), which is why I developed this routing algorithm based on Reinforcement Learning that seeks to maintain a decongested network.

The algorithm is in charge of finding the shortest route for mouse flows (green) and the most uncongested route for elephant flows (reds), the value of the throuput of each link between network switches is known and also if the flow (set of network packets with the same IP and protocol) is an elephant or a mouse.

![routing](https://media.giphy.com/media/Ula6PiO5S7jzlNvL2l/giphy.gif)

## REWARD PER EPISODES

As the Router gain experience with the number of episodes, accuracy and rewards begin to rise. It is possible to achieve an accuracy of 90% given the simplicity of the assembly, for a future project it is expected to take into account many more variables, perhaps a DNQ agent and a much larger network topology.

![Figure 2022-03-15 015334](https://user-images.githubusercontent.com/60159274/158323359-e3d7f5fa-42e8-4d07-90bf-0a29d8112843.png)
