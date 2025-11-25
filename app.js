<<<<<<< HEAD
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
=======
const CONTRACT_ADDRESS = "0x67d06c0F4a20c7CbBd7B3a46F0eFA86d0Ff622F6";

const ABI = [
  "function nextRecipeId() view returns (uint256)",
  "function traceRecipe(uint256 id) view returns (tuple(string recipeName, string description, uint256 timestamp, address chef) recipe, tuple(string name, uint256 quantity, address supplier, uint256 timestamp)[] ingredients)",
  "function addSupplier(address supplier) external",
  "function addChef(address chef) external",
  "function receiveIngredient(string memory _name, uint256 _quantity) external",
  "function produceRecipe(string memory _recipeName, string memory _description, uint256[] memory _ingredientIdsUsed) external"
];

let provider, signer, contract;

async function connect() {
  if (!window.ethereum) return alert("MetaMask required!");
  await window.ethereum.request({ method: "eth_requestAccounts" });
  provider = new ethers.providers.Web3Provider(window.ethereum);
  signer = provider.getSigner();
  contract = new ethers.Contract(CONTRACT_ADDRESS, ABI, signer);

  const address = await signer.getAddress();
  document.getElementById("account").innerText = `Connected: ${address}`;
  document.getElementById("contractSection").classList.remove("hidden");
  loadAllRecipes();
}

async function loadAllRecipes() {
  const div = document.getElementById("recipes");
  div.innerHTML = "Loading from Sepolia...";

  try {
    const nextId = await contract.nextRecipeId();
    let html = "";

    if (nextId.eq(0)) {
      html = "<p>No recipes yet — be the first!</p>";
    } else {
      for (let i = 1; i <= nextId; i++) {
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
    div.innerHTML = "<p>Contract connected but no recipes yet — be the first!</p>";
  }
}

// Victory Toast Button
function addVictoryButton() {
  if (document.getElementById("victoryBtn")) return;
  const button = document.createElement("button");
  button.id = "victoryBtn";
  button.innerText = "Produce Victory Toast + 1000× Champagne (One Click)";
  button.style.backgroundColor = "red";
  button.style.color = "white";
  button.style.padding = "15px 30px";
  button.style.margin = "20px auto";
  button.style.display = "block";
  button.style.fontSize = "16px";
  button.style.borderRadius = "10px";
  button.style.cursor = "pointer";
  button.onclick = produceVictoryToast;
  document.getElementById("contractSection").appendChild(button);
}

async function produceVictoryToast() {
  if (!contract) return alert("Connect wallet first!");
  try {
    const nextId = await contract.nextRecipeId();
    if (!nextId.eq(0)) return alert("Already produced!");
    const addr = await signer.getAddress();
    const tx1 = await contract.addSupplier(addr);
    await tx1.wait();
    const tx2 = await contract.addChef(addr);
    await tx2.wait();
    const tx3 = await contract.receiveIngredient("Champagne", 1000);
    await tx3.wait();
    const tx4 = await contract.produceRecipe("Victory Toast", "Rowan's first on-chain champagne — Nov 2025", [1]);
    await tx4.wait();
    alert("Victory Toast produced! Refresh to see it.");
    loadAllRecipes();
  } catch (err) {
    alert("Error: " + err.message);
  }
}

// Event listeners
document.getElementById("connect").onclick = connect;
document.getElementById("refresh").onclick = loadAllRecipes;

// Add button after connect
document.getElementById("connect").addEventListener("click", addVictoryButton);
>>>>>>> ec34cac (Add one-click Victory Toast button)
