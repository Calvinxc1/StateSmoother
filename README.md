# State Smoother
The State Smoother came out of my frustration with time series smoothing techniques. I've always been very displeased with moving averages, and though exponential smoothing techniques are fantastic, I found the parameters really more of a scattershot than anything. I wanted something with parameters I could have not only finer control over, but had parameters that actually made sense! So after a few years of wrestling with the problem I landed on a rather nice solution, which is this package.

I use this package heavily in my work on New Eden Analytics (of which several of my repos are dedicated to), where I smooth market history data to get a cleaner signal on how prices are moving in the game [EVE Online](https://www.eveonline.com/), which I have been playing for over 15 years. I've also used earlier versions of this package professionally, when dealing with time series data.

## Installation
Working on a pypi version of this, but for now just running `pip install .` from the root of a cloned copy of the reop will do the trick.

## Code Example
The package is very simple to use, and can be seen in the [demonstration notebook](./demonstration.ipynb).

## Roadmap
Located [here](./Roadmap.md).


## Version History
Found [here](./VersionHistory.md).

## Contributing
Like with all my projects, I'm always happy to have contributions. I'm pretty relaxed about it, but be aware that I do follow [Vincent Driessen's Git Branching Model](https://nvie.com/posts/a-successful-git-branching-model/), and am very fond of [Semantic Commit Messages](https://seesparkbox.com/foundry/semantic_commit_messages). So keep that in mind when you're working on a contribution.

One note: I don't care much for linters, I find they organize code in ways I can't easily read. Please don't pass any code you add through a linter unless you are just horrible at writing clean code. And if you are, may I recommend [this book](https://www.amazon.com/gp/product/0132350882/)?

## License
This project uses the [GNU General Public License](./LICENSE).

Short version: Have fun and use it for whatever, just make sure to attribute me for it (-: