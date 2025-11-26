const CONTRACT_ADDRESS = "0x67d06c0F4a20c7CbBd7B3a46F0eFA86d0Ff622F6";

const ABI = [
  "function nextRecipeId() view returns (uint256)",
  "function traceRecipe(uint256) view returns (tuple(string,string,uint256,address), tuple(string,uint256,address,uint256)[])",
  "function addSupplier(address) external",
  "function addChef(address) external",
  "function receiveIngredient(string,uint256) external",
  "function produceRecipe(string,string,uint256[]) external"
];

let provider, signer, contract;

async function connect() {
  if (!window.ethereum) return alert("Install MetaMask!");
  await window.ethereum.request({ method: "eth_requestAccounts" });
  provider = new ethers.providers.Web3Provider(window.ethereum);
  signer = provider.getSigner();
  contract = new ethers.Contract(CONTRACT_ADDRESS, ABI, signer);
  document.getElementById("account").innerText = "Connected: " + await signer.getAddress();
  document.getElementById("contractSection").classList.remove("hidden");
  addVictoryButton();
  loadAllRecipes();
}

async function loadAllRecipes() {
  const div = document.getElementById("recipes");
  div.innerHTML = "Loading...";
  try {
    const nextId = await contract.nextRecipeId();
    if (nextId.eq(0)) {
      div.innerHTML = "<p>Contract connected but no recipes yet — be the first!</p>";
      return;
    }
    let html = "";
    for (let i = 1; i <= nextId; i++) {
      const [recipe, ingredients] = await contract.traceRecipe(i);
      html += `<div class="recipe"><h2>${recipe.recipeName}</h2><p><em>${recipe.description}</em></p><p><small>by ${recipe.chef.slice(0,10)}... • ${new Date(recipe.timestamp*1000).toLocaleString()}</small></p><h4>Ingredients:</h4><ul>${ingredients.map(i=>`<li>${i.name} — ${i.quantity} units</li>`).join("")}</ul></div><hr>`;
    }
    div.innerHTML = html;
  } catch(e) { div.innerHTML = "<p>Error loading recipes</p>"; }
}

function addVictoryButton() {
  if (document.getElementById("victoryBtn")) return;
  const b = document.createElement("button");
  b.id = "victoryBtn";
  b.innerText = "Produce Victory Toast + 1000× Champagne (One Click)";
  b.style.cssText = "background:red;color:white;font-size:18px;padding:16px 32px;border:none;border-radius:12px;cursor:pointer;margin:20px auto;display:block;";
  b.onclick = produceVictoryToast;
  document.getElementById("contractSection").appendChild(b);
}

async function produceVictoryToast() {
  if (!contract) return alert("Connect first!");
  try {
    const addr = await signer.getAddress();
    await (await contract.addSupplier(addr)).wait();
    await (await contract.addChef(addr)).wait();
    await (await contract.receiveIngredient("Champagne",1000)).wait();
    await (await contract.produceRecipe("Victory Toast","Rowan’s first on-chain champagne — Nov 2025",[1])).wait();
    alert("Victory Toast produced! Refresh to see it.");
    loadAllRecipes();
  } catch(e) { alert("Tx failed: "+e.message); }
}

document.getElementById("connect").onclick = connect;
document.getElementById("refresh").onclick = loadAllRecipes;
