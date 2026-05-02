import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { CloudRain, Wind, AlertTriangle, MapPin, Activity } from 'lucide-react';

// Help decide text color based on AQI severity
const getAqiColor = (aqi) => {
  if (aqi <= 50) return 'text-green-400';
  if (aqi <= 100) return 'text-yellow-400';
  if (aqi <= 150) return 'text-orange-400';
  if (aqi <= 200) return 'text-red-400';
  if (aqi <= 300) return 'text-purple-400';
  return 'text-rose-600';
};

const getAqiLabel = (aqi) => {
  if (aqi <= 50) return 'Good';
  if (aqi <= 100) return 'Moderate';
  if (aqi <= 150) return 'Unhealthy for Sensitive Groups';
  if (aqi <= 200) return 'Unhealthy';
  if (aqi <= 300) return 'Very Unhealthy';
  return 'Hazardous';
};

// Full list of cities based on dataset
const AVAILABLE_CITIES = [
  "karachi", "lahore", "beijing", "los angeles", "mumbai", 
  "delhi", "sydney", "london"
];

// Highly reliable Pexels CDN image links
const cityImages = {
  karachi: "https://images.unsplash.com/photo-1602740337312-e28c0b7d27f9?auto=format,compress&w=1920&q=80",
  delhi: "https://images.pexels.com/photos/13708235/pexels-photo-13708235.jpeg?auto=compress&cs=tinysrgb&w=1920&q=80",
  london: "https://images.pexels.com/photos/30754179/pexels-photo-30754179.jpeg?auto=compress&cs=tinysrgb&w=1920&q=80",
  beijing: "https://images.pexels.com/photos/14752024/pexels-photo-14752024.jpeg?auto=compress&cs=tinysrgb&w=1920&q=80",
  "los angeles": "https://images.pexels.com/photos/33280808/pexels-photo-33280808.jpeg?auto=compress&cs=tinysrgb&w=1920&q=80",
  mumbai: "https://images.pexels.com/photos/28684077/pexels-photo-28684077.jpeg?auto=compress&cs=tinysrgb&w=1920&q=80",
  sydney: "https://images.pexels.com/photos/36983267/pexels-photo-36983267.jpeg?auto=compress&cs=tinysrgb&w=1920&q=80",
  lahore: "https://images.pexels.com/photos/14406059/pexels-photo-14406059.jpeg?auto=compress&cs=tinysrgb&w=1920&q=80"
};

export default function App() {
  const [city, setCity] = useState('karachi'); 
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [imgError, setImgError] = useState(false); 
  const baseUrl = import.meta.env.VITE_API_BASE_URL;

  useEffect(() => {
    const fetchForecast = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`${baseUrl}/api/forecast?city=${city}`);
        if (!response.ok) throw new Error('City not found in database or API error');
        
        const jsonData = await response.json();
        setData(jsonData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchForecast();
    setImgError(false); 
  }, [city]);

  return (
    <div className="h-screen w-screen overflow-hidden relative flex items-center justify-center font-sans text-white bg-gray-950">
      
      {/* Background Layer */}
      <img 
        key={city} 
        src={cityImages[city]} 
        alt={`${city} background`}
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-1000 ${imgError ? 'opacity-0' : 'opacity-40'}`}
        onError={() => setImgError(true)} 
      />
      <div className={`absolute inset-0 bg-gradient-to-b from-gray-950 to-cyan-950/20 transition-opacity duration-1000 ${imgError ? 'opacity-100' : 'opacity-0'}`}></div>
      <div className="absolute inset-0 bg-black/40"></div>

      {/* Main Glass Dashboard Card - Tighter padding and locked scrolling */}
      <div className="relative z-10 w-full max-w-5xl px-4 mx-auto">
        <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-3xl p-5 lg:p-6 shadow-[0_0_40px_rgba(0,0,0,0.5)] max-h-[88vh] overflow-hidden flex flex-col">
          
          {/* Header - Reduced margins to save vertical space */}
          <div className="flex flex-col lg:flex-row justify-between items-center mb-4 border-b border-white/10 pb-4 gap-4 shrink-0">
            <div className="w-full lg:w-auto flex flex-col items-center lg:items-start text-center lg:text-left">
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight flex items-center gap-3">
                <Activity className="w-8 h-8 text-cyan-400 shrink-0" />
                AQI Forecast 
              </h1>
              <p className="text-gray-400 mt-4 flex items-center gap-2 text-xs md:text-sm">
                <MapPin className="w-4 h-4 shrink-0" /> Currently Viewing: <span className="font-semibold text-white">{city.charAt(0).toUpperCase() + city.slice(1)}</span>
              </p>
            </div>
            
            {/* Centered Explore Cities Section */}
            <div className="flex flex-col items-center w-full lg:w-1/2">
              <span className="text-[10px] md:text-xs font-bold text-gray-400 mb-2 tracking-widest uppercase">More Cities For You </span>
              <div className="flex flex-wrap justify-center gap-2">
                {AVAILABLE_CITIES.map((c) => (
                  <button
                    key={c}
                    onClick={() => setCity(c)}
                    className={`px-3 py-1.5 md:px-4 md:py-2 rounded-full text-xs md:text-sm font-bold capitalize transition-all duration-300 ${
                      city === c 
                        ? 'bg-cyan-500 text-white shadow-[0_0_15px_rgba(34,211,238,0.4)] border-transparent' 
                        : 'bg-black/50 text-gray-400 border border-white/10 hover:bg-white/10 hover:text-white'
                    }`}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* DYNAMIC CONTENT WRAPPER - Reduced min-height to 350px */}
          <div className="min-h-[350px] flex flex-col justify-center w-full flex-1 min-h-0">
            
            {loading && (
              <div className="flex flex-col items-center justify-center w-full h-full">
                <div className="text-xl md:text-2xl animate-pulse font-semibold tracking-widest text-cyan-400">
                  Judging via Live Data...
                </div>
              </div>
            )}
            
            {error && (
              <div className="flex flex-col items-center justify-center w-full h-full text-red-400">
                <AlertTriangle className="mb-4 w-12 h-12 shrink-0 opacity-80" />
                <h2 className="text-xl font-bold mb-2">Connection Failed</h2>
                <p className="text-sm text-white/70">{error}</p>
              </div>
            )}

            {!loading && !error && data && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6 h-full min-h-0">
                
                {/* Right Now Card - Shrink font and padding */}
                <div className="lg:col-span-1 flex flex-col h-full">
                  <div className="flex-1 p-4 lg:p-5 rounded-2xl bg-black/40 border border-white/10 backdrop-blur-md text-center flex flex-col justify-center items-center shadow-inner">
                    <h2 className="text-xs lg:text-sm tracking-widest text-gray-400 mb-1 uppercase font-bold">Right Now</h2>
                    <div className={`text-5xl lg:text-6xl font-black mb-1 drop-shadow-md ${getAqiColor(data.current_live.aqi)}`}>
                      {data.current_live.aqi}
                    </div>
                    <div className="text-sm lg:text-base font-bold tracking-wide uppercase mb-4 text-gray-200">
                      {getAqiLabel(data.current_live.aqi)}
                    </div>
                    <div className="flex justify-center text-gray-300 text-xs lg:text-sm bg-black/50 border border-white/5 rounded-lg py-2 px-3 w-full">
                      <span className="flex items-center gap-2 font-medium">
                        <CloudRain className="w-3 h-3 lg:w-4 lg:h-4 shrink-0 text-cyan-400"/> PM2.5: {data.current_live.raw_pm25} µg/m³
                      </span>
                    </div>
                  </div>
                </div>

                {/* 3-Day Forecast Section - Shrunk margins and chart height */}
                <div className="lg:col-span-2 flex flex-col justify-between">
                  <div>
                    <h2 className="text-lg lg:text-xl font-bold mb-3 flex items-center gap-2 text-gray-200">
                      <Wind className="w-5 h-5 lg:w-6 lg:h-6 text-cyan-400 shrink-0" />
                      3-Day Machine Learning Forecast
                    </h2>
                    
                    <div className="grid grid-cols-3 gap-3 mb-3">
                      {data.ml_forecast.predictions.map((day) => (
                        <div key={day.date} className="bg-black/40 border border-white/10 rounded-xl p-3 text-center hover:bg-white/5 transition-colors">
                          <div className="text-gray-500 text-[10px] lg:text-xs uppercase tracking-widest mb-0.5 font-bold">{day.date}</div>
                          <div className="font-bold text-sm lg:text-base mb-1 text-gray-300">{day.day_name}</div>
                          <div className={`text-2xl lg:text-3xl font-black drop-shadow-md ${getAqiColor(day.predicted_aqi)}`}>
                            {day.predicted_aqi}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Chart Box - Adjusted height to prevent layout stretching */}
                  <div className="h-36 lg:h-40 w-full bg-black/40 rounded-xl p-3 border border-white/10 shadow-inner mt-auto">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data.ml_forecast.predictions}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="day_name" stroke="#ffffff60" tick={{fill: '#ffffff60', fontSize: 11}} axisLine={false} tickLine={false} />
                        <YAxis stroke="#ffffff60" tick={{fill: '#ffffff60', fontSize: 11}} axisLine={false} tickLine={false} domain={['dataMin - 10', 'dataMax + 10']} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                          itemStyle={{ color: '#22d3ee', fontWeight: 'bold' }} 
                        />
                        <Line 
                          type="monotone" 
                          dataKey="predicted_aqi" 
                          name="Predicted AQI" 
                          stroke="#22d3ee" 
                          strokeWidth={3} 
                          dot={{ r: 4, fill: '#0f172a', stroke: '#22d3ee', strokeWidth: 2 }} 
                          activeDot={{ r: 6, fill: '#22d3ee', stroke: '#fff', strokeWidth: 2 }} 
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}