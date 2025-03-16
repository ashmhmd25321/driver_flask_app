// Additional JavaScript functionality for Flood Prediction System

// Show loading indicator during API calls
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

// Validate day input based on selected month
function validateDayInput() {
    const monthSelect = document.getElementById('month');
    const dayInput = document.getElementById('day');
    
    if (!monthSelect || !dayInput) return;
    
    const month = monthSelect.value;
    const day = parseInt(dayInput.value);
    
    // Define max days for each month
    const maxDays = {
        'January': 31,
        'February': 29, // Accounting for leap years
        'March': 31,
        'April': 30,
        'May': 31,
        'June': 30,
        'July': 31,
        'August': 31,
        'September': 30,
        'October': 31,
        'November': 30,
        'December': 31
    };
    
    // Update max attribute
    dayInput.max = maxDays[month];
    
    // If current value exceeds max, adjust it
    if (day > maxDays[month]) {
        dayInput.value = maxDays[month];
    }
}

// Format date for display
function formatDate(month, day) {
    return `${month} ${day}, 2023`;
}

// Create custom markers with risk level indicators
function createCustomMarker(river, station, coordinates, riskLevel = null) {
    // Default marker options
    const markerOptions = {
        title: `${river} - ${station}`,
        alt: `${river} - ${station}`,
        riseOnHover: true
    };
    
    // Create marker
    const marker = L.marker(coordinates, markerOptions);
    
    // Add risk level to popup if available
    let popupContent = `<b>${river}</b><br>${station}`;
    if (riskLevel) {
        let riskColor = '#28a745'; // Default green for normal
        if (riskLevel === 'HIGH') {
            riskColor = '#dc3545';
        } else if (riskLevel === 'MODERATE') {
            riskColor = '#ffc107';
        }
        
        popupContent += `<br><span style="color:${riskColor}; font-weight:bold;">Risk: ${riskLevel}</span>`;
    }
    
    marker.bindPopup(popupContent);
    
    return marker;
}

// Export data to CSV
function exportToCsv() {
    // Get prediction result data
    const river = document.getElementById('result-river').textContent;
    const station = document.getElementById('result-station').textContent;
    const date = document.getElementById('result-date').textContent;
    const level = document.getElementById('result-level').textContent;
    const risk = document.getElementById('result-risk').textContent.replace(/<[^>]*>/g, ''); // Remove HTML tags
    
    // Create CSV content
    const csvContent = [
        ['River', 'Station', 'Date', 'Predicted Water Level (m)', 'Risk Level'],
        [river, station, date, level, risk]
    ].map(row => row.join(',')).join('\n');
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `flood_prediction_${river}_${station}_${Date.now()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Month change event for day validation
    const monthSelect = document.getElementById('month');
    if (monthSelect) {
        monthSelect.addEventListener('change', validateDayInput);
        // Initial validation
        validateDayInput();
    }
    
    // Add export button event listener
    const exportButton = document.getElementById('export-btn');
    if (exportButton) {
        exportButton.addEventListener('click', exportToCsv);
    }
    
    // Add loading indicators to API calls
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        const originalClickHandler = predictBtn.onclick;
        predictBtn.onclick = function(e) {
            showLoading();
            if (originalClickHandler) {
                originalClickHandler.call(this, e);
            }
        };
    }
}); 