// app.js – Bain Windows & Doors Supply Chain dApp
const CONTRACT_ADDRESS = "0x67d06c0F4a20c7CbBd7B3a46F0eFA86d0Ff622F6";
const ROLES_ADDRESS = "0xd8b934580fcE35a11B58C6D73aDeE468a2833fa8";

const ROLES_ABI = ["function getRole(address) view returns (uint8)"];
const CONTRACT_ABI = [
  "function nextProductId() view returns (uint256)",
  "function products(uint256) view returns (string name, string description, uint8 status, address owner)",
  "function materials(uint256) view returns (string name, uint256 quantity)",
  "function receiveMaterial(string name, uint256 quantity)",
  "function produceProduct(string name, string description, uint256[] materialIds)",
  "function markInstalled(uint256 productId)",
  "function confirmReceipt(uint256 productId)",
  "event MaterialReceived(uint256 id, string name, uint256 quantity)",
  "event ProductProduced(uint256 id, string name)"
];

let provider, signer, contract, rolesContract, userAddress, userRole = 0;

document.getElementById("connect").addEventListener("click", connectWallet);
document.getElementById("refresh")?.addEventListener("click", loadProducts);

async function connectWallet() {
  if (!window.ethereum) return alert("MetaMask not detected!");
  
  try {
    await window.ethereum.request({ method: "eth_requestAccounts" });
    provider = new ethers.providers.Web3Provider(window.ethereum);
    signer = provider.getSigner();
    userAddress = await signer.getAddress();

    contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);
    rolesContract = new ethers.Contract(ROLES_ADDRESS, ROLES_ABI, provider);
    userRole = Number(await rolesContract.getRole(userAddress));

    document.getElementById("account").innerHTML = 
      `Connected: <strong>${userAddress.slice(0,10)}...</strong><br>Role: <strong>${["None","ADMIN","Supplier","Installer","Homeowner"][userRole]}</strong>`;

    document.getElementById("contractSection").classList.remove("hidden");
    showPanels();
    loadProducts();
  } catch (err) {
    alert("Connection failed: " + err.message);
  }
}

function showPanels() {
  document.querySelectorAll(".role-section").forEach(p => p.classList.add("hidden"));
  if (userRole === 1) document.getElementById("adminPanel").classList.remove("hidden");
  if (userRole === 2) document.getElementById("supplierPanel").classList.remove("hidden");
  if (userRole === 3) document.getElementById("installerPanel").classList.remove("hidden");
  if (userRole === 4) document.getElementById("homeownerPanel").classList.remove("hidden");
}

async function loadProducts() {
  const list = document.getElementById("products");
  list.innerHTML = "Loading…";
  try {
    const nextId = await contract.nextProductId();
    if (nextId.eq(0)) { list.innerHTML = "No products yet."; return; }
    let html = "";
    for (let i = 1; i < nextId; i++) {
      const p = await contract.products(i);
      html += `<div class="product"><h3>Product #${i}: ${p.name}</h3><p>${p.description}</p><p>Status: ${["Received","Produced","Installed","Delivered"][p.status]}</p></div>`;
    }
    list.innerHTML = html || "No products found.";
  } catch { list.innerHTML = "Error loading products"; }
}

// Quick admin assign (you’ll use this next)
async function assignRole() {
  const addr = document.getElementById("roleAddr").value;
  const role = document.getElementById("roleNum").value;
  const rolesWithSigner = new ethers.Contract(ROLES_ADDRESS, ["function setRole(address,uint8) external"], signer);
  await (await rolesWithSigner.setRole(addr, role)).wait();
  alert("Role assigned!");
}

// Hook up buttons
document.getElementById("victoryBtn")?.addEventListener("click", () => alert("Victory! (demo button)"));
