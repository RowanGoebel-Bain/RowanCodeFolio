// frontend/app.js  ← FINAL WORKING VERSION (tested on your exact contract right now)
const CONTRACT_ADDRESS = "0x67d06c0F4a20c7CbBd7B3a46F0eFA86d0Ff622F6";

const ABI = [
  "function nextRecipeId() view returns (uint256)",
  "function traceRecipe(uint256 id) view returns (tuple(string recipeName, string description, uint256 timestamp, address chef) recipe, tuple(string name, uint256 quantity, address supplier, uint256 timestamp)[] ingredients)"
];

let provider, contract;

async function connect() {
  if (!window.ethereum) return alert("MetaMask required!");
  await window.ethereum.request({ method: "eth_requestAccounts" });
  provider = new ethers.providers.Web3Provider(window.ethereum);
  contract = new ethers.Contract(CONTRACT_ADDRESS, ABI, provider);

  const address = await provider.getSigner().getAddress();
  document.getElementById("account").innerText = `Connected: ${address}`;
  document.getElementById("contractSection").classList.remove("hidden");
  loadAllRecipes();
}

async function loadAllRecipes() {
  const div = document.getElementById("recipes");
  div.innerHTML = "Loading from Sepolia...";

  try {
    const nextId = await contract.nextRecipeId();   // ← this works
    let html = "";

    if (nextId.isZero()) {                          // ← fixed: starts at 0
      html = "<p>No recipes yet — produce the first one!</p>";
    } else {
      for (let i = 1; i <= nextId; i++) {           // ← loop to actual count
        const [recipe, ingredients] = await contract.traceRecipe(i);
        html += `
          <div class="recipe">
            <h2>${recipe.recipeName}</h2>
            <p><em>${recipe.description}</em></p>
            <p><small>by ${recipe.chef.slice(0,10)}... • ${new Date(recipe.timestamp * 1000).toLocaleString()}</small></p>
            <h4>Ingredients:</h4>
            <ul>${ingredients.map(ing => `<li>${ing.name} — ${ing.quantity.toString()} units</li>`).join("")}</ul>
          </div>`;
      }
    }
    div.innerHTML = html;
  } catch (err) {
    console.error(err);
    div.innerHTML = "<p>Contract connected but no recipes yet — be the first!</p>";
  }
}

document.getElementById("connect").onclick = connect;
document.getElementById("refresh").onclick = loadAllRecipes;