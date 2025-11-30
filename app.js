const CONTRACT_ADDRESS = "0x67d06c0F4a20c7CbBd7B3a46F0eFA86d0Ff622F6";
const ROLES_ADDRESS = "0xd8b934580fcE35a11B58C6D73aDeE468a2833fa8";

const ABI = [
  "function nextRecipeId() view returns (uint256)",
  "function traceRecipe(uint256) view returns (tuple(string,string,uint256,address), tuple(string,uint256,address,uint256)[])",
  "function addSupplier(address) external",
  "function addChef(address) external",  // Repurpose as addInstaller if needed
  "function receiveIngredient(string,uint256) external",
  "function produceRecipe(string,string,uint256[]) external"
  // Add more if contract updated: e.g., "function markInstalled(uint256) external", "function confirmReceipt(uint256) external"
];

const ROLES_ABI = ["function getRole(address) view returns (uint8)", "function setRole(address,uint8) external"];

let provider, signer, contract, rolesContract, userRole = 0;

async function connect() {
  toggleLoading(true);
  try {
    await ethereum.request({ method: "eth_requestAccounts" });
    provider = new ethers.providers.Web3Provider(window.ethereum);
    signer = provider.getSigner();
    const addr = await signer.getAddress();

    contract = new ethers.Contract(CONTRACT_ADDRESS, ABI, signer);
    rolesContract = new ethers.Contract(ROLES_ADDRESS, ROLES_ABI, signer);

    userRole = Number(await rolesContract.getRole(addr));

    document.getElementById("account").innerHTML = 
      `<strong>Connected:</strong> ${addr.slice(0, 10)}...<br><strong>Role:</strong> ${["None", "Admin", "Supplier", "Installer", "Homeowner"][userRole]}`;

    document.getElementById("contractSection").classList.remove("hidden");
    showRolePanels(userRole);
    loadAllProducts();
  } catch (error) {
    console.error(error);
    alert("Error connecting to wallet.");
  } finally {
    toggleLoading(false);
  }
}

function showRolePanels(role) {
  document.querySelectorAll(".role-section").forEach(el => el.style.display = "none");
  if (role === 1) document.getElementById("adminPanel").style.display = "block";
  if (role === 2) document.getElementById("supplierPanel").style.display = "block";
  if (role === 3) document.getElementById("installerPanel").style.display = "block";
  if (role === 4) document.getElementById("homeownerPanel").style.display = "block";
}

async function loadAllProducts() {
  const div = document.getElementById("products");
  div.innerHTML = "";
  toggleLoading(true);
  try {
    const n = await contract.nextRecipeId();
    if (n.eq(0)) return div.innerHTML = "No products available yet.";

    let html = "";
    for (let i = 1; i <= n; i++) {
      const [product, materials] = await contract.traceRecipe(i);
      html += `
        <div class="product">
          <h2>${product.recipeName}</h2>  <!-- Rename to productName in contract if possible -->
          <p>${product.description}</p>
          <h4>Materials:</h4>
          <ul>
            ${materials.map(m => `<li>${m[0]} (${m[1]} units) - Supplier: ${m[2].slice(0,10)}... - Timestamp: ${new Date(Number(m[3]) * 1000).toLocaleString()}</li>`).join('')}
          </ul>
        </div>
      `;
    }
    div.innerHTML = html || "No products found.";
  } catch (e) {
    console.error(e);
    div.innerHTML = "Error loading products.";
  } finally {
    toggleLoading(false);
  }
}

async function assignRole() {
  const addr = document.getElementById("roleAddr").value;
  const role = document.getElementById("roleNum").value;
  try {
    const tx = await rolesContract.setRole(addr, role);
    await tx.wait();
    alert("Role assigned!");
  } catch (e) {
    alert("Error assigning role: " + e.message);
  }
}

async function receiveMaterial() {
  const name = document.getElementById("materialName").value;
  const qty = document.getElementById("materialQty").value;
  try {
    const tx = await contract.receiveIngredient(name, qty);
    await tx.wait();
    alert("Material received!");
    loadAllProducts();
  } catch (e) {
    alert("Error: " + e.message);
  }
}

async function produceProduct() {
  const name = document.getElementById("productName").value;
  const desc = document.getElementById("productDesc").value;
  const ids = document.getElementById("materialIds").value.split(",").map(id => parseInt(id.trim()));
  try {
    const tx = await contract.produceRecipe(name, desc, ids);
    await tx.wait();
    alert("Product produced!");
    loadAllProducts();
  } catch (e) {
    alert("Error: " + e.message);
  }
}

// Placeholder for Installer (add to contract if needed)
async function markInstalled() {
  const id = document.getElementById("productIdInstall").value;
  // Assuming contract has markInstalled(uint256)
  try {
    const tx = await contract.markInstalled(id);
    await tx.wait();
    alert("Installation marked!");
    loadAllProducts();
  } catch (e) {
    alert("Error: " + e.message);
  }
}

// Placeholder for Homeowner (add to contract if needed)
async function confirmReceipt() {
  const id = document.getElementById("productIdConfirm").value;
  // Assuming contract has confirmReceipt(uint256)
  try {
    const tx = await contract.confirmReceipt(id);
    await tx.wait();
    alert("Receipt confirmed!");
    loadAllProducts();
  } catch (e) {
    alert("Error: " + e.message);
  }
}

async function produceVictoryProduct() {
  try {
    // Hardcode a sample: e.g., assume material IDs [1,2] exist
    const tx = await contract.produceRecipe("Victory Window", "Premium window with 1000x frames", [1, 2]);  // Adjust IDs
    await tx.wait();
    alert("Victory product produced!");
    loadAllProducts();
  } catch (e) {
    alert("Error: " + e.message);
  }
}

function toggleLoading(isLoading) {
  const spinner = document.getElementById("loadingSpinner");
  const productsDiv = document.getElementById("products");
  spinner.style.display = isLoading ? "block" : "none";
  productsDiv.style.display = isLoading ? "none" : "block";
}

document.getElementById("connect").onclick = connect;
document.getElementById("refresh").onclick = loadAllProducts;
